import math
import os
import numpy as np
import pandas as pd
from typing import Tuple
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import gamma as gamma_dist, gaussian_kde
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
import warnings
from plotly.subplots import make_subplots
from functools import lru_cache

# more real data for delay disribution
# https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024?utm_source=chatgpt.com
# https://datacatalog.med.nyu.edu/dataset/10438?utm_source=chatgpt.com

warnings.filterwarnings("ignore", category=FutureWarning)

from functools import lru_cache

@lru_cache(maxsize=1)
def load_cases_df():
    df = pd.read_csv("./data/combined_state_no_revision.csv")
    # downcast + normalize
    df["time_value"] = pd.to_datetime(df["time_value"])
    df = df.dropna(subset=["JHU-Cases"])
    df["geo_value"] = df["geo_value"].astype("category")
    # float32 saves memory
    df["JHU-Cases"] = pd.to_numeric(df["JHU-Cases"], errors="coerce", downcast="float")
    df = df.sort_values("time_value")
    return df

@lru_cache(maxsize=1)
def get_geo_options():
    df = load_cases_df()
    return tuple(sorted(df["geo_value"].cat.categories.tolist()))

@lru_cache(maxsize=1)
def prebuild_series_by_geo():
    """
    Build a dict: geo -> (time_numpy, cases_numpy)
    All arrays are float32 to reduce RAM.
    """
    df = load_cases_df()
    # ensure daily unique index per geo; your groupby-mean can be done once here
    g = df.groupby(["geo_value", "time_value"])["JHU-Cases"].mean().rename("cases").reset_index()
    out = {}
    for geo, sub in g.groupby("geo_value"):
        sub = sub.sort_values("time_value")
        out[geo] = (
            sub["time_value"].to_numpy(),                    # datetime64[ns]
            sub["cases"].to_numpy(dtype=np.float32)          # float32
        )
    return out

REAL_SOURCES = {
    "cn30": "./data/cn30_linelist_reporting_delay_from_symptom_onset.csv",
    "uscdc": "./data/uscdc_linelist_reporting_delay_from_symptom_onset.csv",
    "hk": "./data/hk_linelist_reporting_delay_from_symptom_onset.csv",
}

@lru_cache(maxsize=8)
def load_real_delay_source(kind: str) -> pd.DataFrame:
    path = REAL_SOURCES.get(kind)
    if path is None:
        raise ValueError(f"Unknown real-source: {kind}")
    df = pd.read_csv(path)
    df["reference_date"] = pd.to_datetime(df["reference_date"])
    df["report_date"] = pd.to_datetime(df.get("report_date"), errors="coerce")
    # downcast
    df["count"] = pd.to_numeric(df["count"], errors="coerce", downcast="float")
    df["delay"] = pd.to_numeric(df["delay"], errors="coerce", downcast="integer").astype(int)
    df = df[df["delay"] >= 0].copy()
    if kind == "uscdc":
        df = df[df["delay"] > 0].copy()
    return df

# =============== Rolling kernel builder (Histogram or KDE) ===============
def build_delay_kernel_for_window(
    df_delays: pd.DataFrame,
    window_end: pd.Timestamp,
    window_days: int = 30,
    max_delay: int = 60,
    method: str = "hist",  # 'hist' or 'kde'
) -> Tuple[np.ndarray, int]:
    """
    Make a discrete delay kernel on grid 0..max_delay for the window (t-Y, t].
    Returns (kernel, case_count_in_window)
    """
    t0 = window_end - pd.Timedelta(days=window_days)
    window = df_delays[(df_delays["reference_date"] > t0) & (df_delays["reference_date"] <= window_end)]
    if window.empty:
        return np.zeros(max_delay + 1, dtype=float), 0

    # Aggregate within window by delay
    agg = window.groupby("delay", as_index=False)["count"].sum()
    agg = agg.sort_values("delay")

    # Limit to [0, max_delay]
    agg = agg[agg["delay"] <= max_delay]
    if agg.empty:
        return np.zeros(max_delay + 1, dtype=float), 0

    grid = np.arange(0, max_delay + 1)

    if method == "hist":
        hist = np.zeros_like(grid, dtype=float)
        delays = agg["delay"].to_numpy()
        counts = agg["count"].to_numpy()
        hist[delays] = counts
        total = hist.sum()
        if total > 0:
            hist = hist / total
        return hist, int(counts.sum())

    # KDE
    kde = gaussian_kde(agg["delay"].to_numpy(), weights=agg["count"].to_numpy())
    pdf = kde(grid)
    pdf = np.clip(pdf, 0, None)
    s = pdf.sum()
    if s > 0:
        pdf = pdf / s
    return pdf, int(agg["count"].sum())

# =============== Method-of-moments Gamma fit from weighted delays =========
def gamma_moments_from_window(df_delays: pd.DataFrame, window_end: pd.Timestamp, window_days: int = 30):
    t0 = window_end - pd.Timedelta(days=window_days)
    window = df_delays[(df_delays["reference_date"] > t0) & (df_delays["reference_date"] <= window_end)]
    if window.empty:
        return np.nan, np.nan, 0, np.nan, np.nan

    agg = window.groupby("delay", as_index=False)["count"].sum()
    w = agg["count"].to_numpy(dtype=float)
    x = agg["delay"].to_numpy(dtype=float)
    N = w.sum()
    if N <= 0:
        return np.nan, np.nan, 0, np.nan, np.nan
    mu = (w * x).sum() / N
    m2 = (w * (x ** 2)).sum() / N
    var = m2 - mu ** 2
    if mu <= 0 or var <= 0:
        return mu, var, int(N), np.nan, np.nan
    k = mu ** 2 / var
    theta = var / mu
    return mu, var, int(N), k, theta

# =============== Synthetic Gamma delay kernel =============================
def get_gamma_delay(length: int, mean: float, scale: float) -> np.ndarray:
    x = np.arange(length)
    a = mean / scale
    delay = gamma_dist.pdf(x, a=a, scale=scale)
    s = delay.sum()
    if s > 0:
        delay = delay / s
    return delay

# =============== Deconvolution Methods ===================================
def fft_deconvolution(observed, delay, eps=1e-3):
    padded_delay = np.zeros(len(observed))
    padded_delay[:len(delay)] = delay
    obs_fft = fft(observed)
    delay_fft = fft(padded_delay)
    delay_fft[np.abs(delay_fft) < eps] = eps
    recon_fft = obs_fft / delay_fft
    recon = np.real(ifft(recon_fft))
    recon[recon < 0] = 0
    return recon

def wiener_deconvolution(observed, delay, snr=10):
    padded_delay = np.zeros(len(observed))
    padded_delay[:len(delay)] = delay
    obs_fft = fft(observed)
    delay_fft = fft(padded_delay)
    power_H = np.abs(delay_fft) ** 2
    recon_fft = np.conj(delay_fft) * obs_fft / (power_H + 1.0 / snr)
    recon = np.real(ifft(recon_fft))
    recon[recon < 0] = 0
    return recon

# =============== NEW: Forward model (reconvolution) + RMSE ===============
def reconvolve_linear(signal, kernel, out_len=None):
    """
    Linear (non-circular) convolution of signal with kernel, truncated to out_len.
    Kernel is normalized to sum to 1 so scales match the observed series.
    """
    if out_len is None:
        out_len = len(signal)
    k = np.array(kernel, dtype=float)
    ks = k.sum()
    if ks > 0:
        k = k / ks
    y_full = np.convolve(np.asarray(signal, dtype=float), k, mode="full")
    return y_full[:out_len]

def rmse(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def estimate_lambda_from_residuals(confirmed, reconv):
    """
    Returns lambda ≈ 1/SNR using residuals:
      SNR ≈ Var(signal) / Var(noise) with
      signal ≈ reconv, noise ≈ confirmed - reconv
    """
    confirmed = np.asarray(confirmed, float)
    reconv = np.asarray(reconv, float)
    resid = confirmed - reconv

    # Sample variances (bias doesn't matter for a ratio)
    var_signal = float(np.var(reconv, ddof=1)) if len(reconv) > 1 else 0.0
    var_noise  = float(np.var(resid,  ddof=1)) if len(resid)  > 1 else 0.0

    if var_signal <= 0 or var_noise <= 0:
        # Fallback (very conservative)
        return 1.0
    snr = var_signal / var_noise
    return 1.0 / snr  # lambda ≈ 1/SNR

def tikhonov_deconvolution(observed, delay, lam=0.1):
    """
    Tikhonov (ridge) deconvolution:
      I_hat(f) = conj(G(f)) / (|G(f)|^2 + lam) * C(f)
    """
    n = len(observed)
    padded_delay = np.zeros(n)
    padded_delay[:len(delay)] = delay

    obs_fft = fft(observed)
    G = fft(padded_delay)
    filt = np.conj(G) / (np.abs(G)**2 + lam)

    recon = np.real(ifft(filt * obs_fft))
    recon[recon < 0] = 0
    return recon

# =============== Dash App UI =============================================
def create_app(server, prefix="/app_delay_filtering/"):
    dash_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix=prefix,
        requests_pathname_prefix=prefix,
        suppress_callback_exceptions=True,
        title="COVID Delay-Aware Deconvolution (Rolling Kernel)"
    )
    geo_options = get_geo_options()

    dash_app.layout = html.Div([
        html.Div([
            html.H2("Reverse Confirmed Cases to Symptomatic Curve"),
            html.Label("Select Region:"),
            dcc.Dropdown(id="geo-dropdown",
                         options=[{"label": g, "value": g} for g in geo_options],
                         value=geo_options[0]),
            html.Br(),
            html.Label("Delay Source:"),
            dcc.Dropdown(
                id="delay-source",
                options=[
                    {"label": "Synthetic (Gamma)", "value": "synthetic"},
                    {"label": "Real (30 Provinces in China linelist)", "value": "cn30"},
                    {"label": "Real (US CDC linelist)", "value": "uscdc"},
                    {"label": "Real (Hong Kong linelist)", "value": "hk"},
                ],
                value="synthetic"
            ),
            html.Div(id="real-source-info", style={"color": "blue", "marginTop": "5px"}),

            html.Div([
                html.Label("Gamma Mean:"),
                dcc.Input(id="mean", type="number", value=5, step=0.1),
                html.Br(), html.Br(),
                html.Label("Gamma Scale:"),
                dcc.Input(id="scale", type="number", value=1, step=0.1),
            ], id="gamma-controls", style={"marginTop": "8px"}),

            html.Div([
                html.Label("Kernel Method:"),
                dcc.RadioItems(
                    id="kernel-method",
                    options=[
                        {"label": "Histogram", "value": "hist"},
                        {"label": "KDE (weighted)", "value": "kde"}
                    ],
                    value="hist",
                    labelStyle={"display": "inline-block", "marginRight": "10px"}
                ),
                html.Br(),
                html.Label("Window length (days, Y):"),
                dcc.Input(id="window-days", type="number", value=30, min=1, step=1),
                html.Br(), html.Br(),
                html.Label("Max delay to consider (days):"),
                dcc.Input(id="max-delay", type="number", value=60, min=1, step=1),
                html.Br(), html.Br(),
                html.Label("Window end date (t):"),
                dcc.DatePickerSingle(id="window-end",
                                     display_format="YYYY-MM-DD",
                                     placeholder="YYYY-MM-DD")
            ], id="real-controls", style={"display": "none", "marginTop": "8px"}),

            html.Br(),
            html.Label("Amplitude of the Extra High Freq Component"),
            dcc.Slider(id="hf-strength", min=0, max=3, step=0.1, value=0.5,
                       tooltip={"placement": "bottom"},
                       marks={0: "0", 1: "1", 2: "2", 3: "3"},
                       included=False),
            html.Br(),
            html.Label("Freq (cycles/day) of the Extra High Freq Component"),
            dcc.Slider(id="hf-freq", min=0.05, max=0.5, step=0.01, value=0.25,
                       tooltip={"placement": "bottom"},
                       marks={0.05: "0.05", 0.25: "0.25", 0.5: "0.5"},
                       included=False),
            html.Br(),
            html.Label("Influence threshold 𝑰_th"),
            dcc.Slider(
                id="influence-thresh",
                min=0.0, max=1.0, step=0.01, value=0.10,
                tooltip={"placement": "bottom"},
                marks={0: "0.00", 0.1: "0.10", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1.0: "1.00"},
                included=False
            ),
            html.Br(),
            html.Button("Update", id="update-btn", n_clicks=0),
        ], style={"width": "25%", "padding": "20px"}),

        html.Div([
            dcc.Graph(id="result-plot", style={"height": "400px"}),
            html.Div([
                dcc.Graph(id="delay-plot", style={"width": "45%", "height": "380px", "display": "inline-block"}),
                dcc.Graph(id="fft-plot", style={"width": "53%", "height": "380px", "display": "inline-block"})
            ])
        ], style={"width": "75%", "padding": "20px"})
    ], style={"display": "flex", "flexDirection": "row", "flexWrap": "nowrap", "width": "100vw"})

    # ------- Callbacks (注意把装饰器对象改为 dash_app) -------
    @dash_app.callback(
        Output("gamma-controls", "style"),
        Output("real-controls", "style"),
        Input("delay-source", "value")
    )
    def toggle_controls(delay_source):
        if delay_source == "synthetic":
            return {"display": "block"}, {"display": "none"}
        else:
            return {"display": "none"}, {"display": "block"}

    @dash_app.callback(
        Output("real-source-info", "children"),
        Input("delay-source", "value")
    )
    def update_real_source_info(delay_source):
        if delay_source == "synthetic":
            return ""
        try:
            df_real = load_real_delay_source(delay_source)
            total_count = df_real["count"].sum()
            min_date = df_real["reference_date"].min().date()
            max_date = df_real["reference_date"].max().date()
            return f"Total counts: {int(total_count)}, available window end date: {min_date} to {max_date}"
        except Exception as e:
            return f"Error loading real source: {e}"

    @dash_app.callback(
        Output("result-plot", "figure"),
        Output("delay-plot", "figure"),
        Output("fft-plot", "figure"),
        Input("update-btn", "n_clicks"),
        State("geo-dropdown", "value"),
        State("delay-source", "value"),
        State("mean", "value"),
        State("scale", "value"),
        State("kernel-method", "value"),
        State("window-days", "value"),
        State("max-delay", "value"),
        State("window-end", "date"),
        State("hf-strength", "value"),
        State("hf-freq", "value"),
        State("influence-thresh", "value"),
    )
    def update_figures(n_clicks, geo_value, delay_source, mean, scale,
                       kernel_method, window_days, max_delay, window_end_str,
                       hf_strength, hf_freq, I_thresh=0.1):
        series_map = prebuild_series_by_geo()

        if geo_value not in series_map:
            return go.Figure(), go.Figure(), go.Figure()  # no data fallback

        time, confirmed = series_map[geo_value]
        confirmed = confirmed.astype(np.float64, copy=False)  # ensure numeric

        if delay_source == "synthetic":
            L = min(len(confirmed), (max_delay if max_delay else 60) + 1)
            delay_kernel = get_gamma_delay(L, mean, scale)
            label = f"Gamma(mean={mean}, scale={scale})"
            mu = np.sum(np.arange(L) * delay_kernel)
            var = np.sum((np.arange(L) ** 2) * delay_kernel) - mu**2
            gamma_fit_txt = f" (μ≈{mu:.2f}, σ²≈{var:.2f})"
        else:
            real_df = load_real_delay_source(delay_source)
            window_end = real_df["reference_date"].max() if window_end_str is None else pd.to_datetime(window_end_str)
            Y = int(window_days) if window_days else 30
            Dmax = int(max_delay) if max_delay else 60
            delay_kernel, n_cases = build_delay_kernel_for_window(
                real_df, window_end=window_end, window_days=Y, max_delay=Dmax, method=kernel_method
            )
            mu, var, _, k, theta = gamma_moments_from_window(real_df, window_end=window_end, window_days=Y)
            src_label = delay_source.upper()
            date_range_txt = f"available [{real_df['reference_date'].min().date()}, {real_df['reference_date'].max().date()}]"
            if not np.isnan(k):
                gamma_fit_txt = f"  (MoM Gamma: k={k:.2f}, θ={theta:.2f}, μ={mu:.2f}, σ²={var:.2f})"
            else:
                gamma_fit_txt = f"  (μ={mu:.2f}, σ²={var:.2f}, MoM Gamma unavailable)"
            label = f"{src_label} {kernel_method.upper()} — window (t-{Y}, t], t={window_end.date()}, N={n_cases}, {date_range_txt}"

        if len(delay_kernel) > len(confirmed):
            delay_kernel = delay_kernel[:len(confirmed)]
            delay_kernel = delay_kernel / delay_kernel.sum() if delay_kernel.sum() > 0 else delay_kernel

        wiener_curve = wiener_deconvolution(confirmed, delay_kernel)

        freq = hf_freq
        amp = hf_strength * np.max(wiener_curve)
        t = np.arange(len(wiener_curve))
        highfreq_signal = amp * np.sin(2 * np.pi * freq * t)
        altered_infection = wiener_curve + highfreq_signal

        wiener_reconv = reconvolve_linear(wiener_curve, delay_kernel, out_len=len(confirmed))
        altered_reconv = reconvolve_linear(altered_infection, delay_kernel, out_len=len(confirmed))

        rmse_wiener = rmse(confirmed, wiener_reconv)
        rmse_altered = rmse(confirmed, altered_reconv)

        lambda_tikh = estimate_lambda_from_residuals(confirmed, wiener_reconv)
        tikhonov_curve = tikhonov_deconvolution(confirmed, delay_kernel, lam=lambda_tikh)
        tikhonov_reconv = reconvolve_linear(tikhonov_curve, delay_kernel, out_len=len(confirmed))
        rmse_tikh = rmse(confirmed, tikhonov_reconv)

        delay_fft_complex = fft(delay_kernel)
        A = np.abs(delay_fft_complex)
        A2 = A**2
        freqs = fftfreq(len(delay_kernel), d=1.0)
        pos = freqs > 0

        lambda_base = estimate_lambda_from_residuals(confirmed, wiener_reconv)
        lambda_alt  = estimate_lambda_from_residuals(confirmed, altered_reconv)
        lambda_tikh2 = estimate_lambda_from_residuals(confirmed, tikhonov_reconv)

        H_tikh = A2 / (A2 + lambda_tikh2)
        influence_base = A2 / (A2 + lambda_base)
        influence_alt  = A2 / (A2 + lambda_alt)

        f_cut_base = f_cut_alt = f_cut_tikh = None
        P_cut_base = P_cut_alt = P_cut_tikh = None
        if np.any(pos):
            fpos = freqs[pos]
            def first_cut(I):
                idx = np.where(I[pos] < I_thresh)[0]
                if idx.size > 0:
                    f = float(fpos[idx[0]])
                    return f, (1.0 / f) if f > 0 else np.inf
                return None, None
            f_cut_base, P_cut_base = first_cut(influence_base)
            f_cut_alt,  P_cut_alt  = first_cut(influence_alt)
            f_cut_tikh, P_cut_tikh = first_cut(H_tikh)

        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=time, y=confirmed, mode='lines', name='Confirmed Cases'))
        fig_main.add_trace(go.Scatter(x=time, y=wiener_curve, mode='lines', name='Wiener deconv Symptomatic'))
        fig_main.add_trace(go.Scatter(x=time, y=wiener_reconv, mode='lines', line=dict(dash='dash'),
                                      name=f'Wiener re-conv (RMSE {rmse_wiener:.1f})'))
        fig_main.add_trace(go.Scatter(x=time, y=altered_infection, mode='lines', name='Alt Symptomatic (+hi-freq)'))
        fig_main.add_trace(go.Scatter(x=time, y=altered_reconv, mode='lines', line=dict(dash='dot'),
                                      name=f'Alt reconv (RMSE {rmse_altered:.1f})'))
        fig_main.add_trace(go.Scatter(x=time, y=tikhonov_curve, mode='lines',
                                      line=dict(color='green'),
                                      name=f'Tikhonov deconv Symptomatic (λ={lambda_tikh:.3g})'))
        fig_main.add_trace(go.Scatter(x=time, y=tikhonov_reconv, mode='lines',
                                      line=dict(color='green', dash='dash'),
                                      name=f'Tikhonov re-conv (RMSE {rmse_tikh:.1f})'))
        fig_main.update_layout(title=f"US Infection Reconstruction in {geo_value.upper()} — The raw vs. Wiener vs. Altered",
                               xaxis_title="Date", yaxis_title="Cases", height=400,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5),
                               margin=dict(t=100))

        grid = np.arange(len(delay_kernel))
        fig_delay = go.Figure()
        fig_delay.add_trace(go.Scatter(x=grid, y=delay_kernel, mode='lines', name="Delay kernel"))
        fig_delay.update_layout(title=f"Delay Distribution<br> — {label}{gamma_fit_txt}",
                                xaxis_title="Delay (days)", yaxis_title="Probability", height=400)

        fig_fft = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fft.add_trace(go.Scatter(x=freqs[pos], y=A2[pos], mode='lines', name="Delay power |G(f)|²"),
                          secondary_y=False)
        fig_fft.add_trace(go.Scatter(x=freqs[pos], y=influence_base[pos], mode='lines',
                                     name="Influence I(f) — baseline", line=dict(color="red")),
                          secondary_y=True)
        fig_fft.add_trace(go.Scatter(x=freqs[pos], y=influence_alt[pos], mode='lines',
                                     name="Influence I(f) — altered", line=dict(color="orange")),
                          secondary_y=True)
        fig_fft.add_trace(go.Scatter(x=freqs[pos], y=H_tikh[pos], mode='lines',
                                     name="Tikhonov S(f)", line=dict(color="green")),
                          secondary_y=True)

        def add_cut(fig, f_val, color, label):
            if f_val is None or not np.isfinite(f_val): return
            fig.add_trace(go.Scatter(x=[f_val, f_val], y=[0, 1], mode="lines",
                                     line=dict(color=color, dash="dash", width=2),
                                     name=label, yaxis="y2"), secondary_y=True)

        if f_cut_base: add_cut(fig_fft, f_cut_base, "red",   f"f_cut(base)≈{f_cut_base:.3f} (P≈{P_cut_base:.1f}d)")
        if f_cut_alt:  add_cut(fig_fft, f_cut_alt,  "orange",f"f_cut(alt)≈{f_cut_alt:.3f} (P≈{P_cut_alt:.1f}d)")
        if f_cut_tikh: add_cut(fig_fft, f_cut_tikh, "green", f"f_cut(tikh)≈{f_cut_tikh:.3f} (P≈{P_cut_tikh:.1f}d)")

        fig_fft.update_layout(
            title="Delay Spectrum & SNR-aware Influence",
            height=400,
            legend=dict(orientation="h", yanchor="top", y=-0.7, xanchor="center", x=0.5),
            margin=dict(t=80, b=80)
        )
        details_lines = [f"λ_base≈{lambda_base:.3g}, λ_alt≈{lambda_alt:.3g}, λ_tikh≈{lambda_tikh2:.3g}"]
        fig_fft.add_annotation(x=0.5, y=1.35, xref="paper", yref="paper",
                               text="<br>".join(details_lines), showarrow=False,
                               xanchor="center", font=dict(size=15))
        fig_fft.update_xaxes(title_text="Frequency (cycles/day)")
        fig_fft.update_yaxes(title_text="Power |G(f)|²", secondary_y=False)
        fig_fft.update_yaxes(title_text="Influence I(f)", range=[0, 1], secondary_y=True)

        return fig_main, fig_delay, fig_fft

    return dash_app