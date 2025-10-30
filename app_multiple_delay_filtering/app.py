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

MINI_H = 360
MINI_MARGIN = dict(t=40, b=30, l=48, r=10)   # extra bottom space for 1-line legend
LEGEND = dict(
    orientation="h",
    x=0.5, xanchor="center",
    y=-0.30, yanchor="top",                 # push legend below the plot area
    font=dict(size=12),
)

# more real data for delay disribution
# https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024?utm_source=chatgpt.com
# https://datacatalog.med.nyu.edu/dataset/10438?utm_source=chatgpt.com

warnings.filterwarnings("ignore", category=FutureWarning)

# =============== Load Confirmed Case Data (memory-friendly) ===============
CASES_PATH = "./data/combined_state_no_revision.csv"

def _load_cases_df():
    df = pd.read_csv(
        CASES_PATH,
        usecols=["time_value", "geo_value", "JHU-Cases"],  # only what we use
        dtype={"geo_value": "category", "JHU-Cases": "float32"},  # downcast
        parse_dates=["time_value"]
    )
    df = df.dropna(subset=["JHU-Cases"]).sort_values("time_value")
    # Optional: pre-aggregate to daily mean per (geo, date) to shrink rows
    df["time_value"] = df["time_value"].dt.normalize()     # strip time part
    df = (df.groupby(["geo_value", "time_value"], observed=True)["JHU-Cases"]
            .mean().reset_index())
    return df

df_full = _load_cases_df()
geo_options = list(df_full["geo_value"].cat.categories)

# =============== Helper: Real delay sources =====================

# ---- Unified color palette ----
COLORS = {
    "confirmed": "#444444",   # deep gray (observed, always)
    "delay":     "#000000",   # black line
    "wiener":    "#4C78A8",   # blue
    "alt":       "#E45756",   # red (you can switch to orange: #F58518)
    "tikhonov":  "#54A24B",   # green
}

REAL_SOURCES = {
    "cn30": "./data/cn30_linelist_reporting_delay_from_symptom_onset.csv",
    "uscdc": "./data/uscdc_linelist_reporting_delay_from_symptom_onset.csv",
    "hk": "./data/hk_linelist_reporting_delay_from_symptom_onset.csv",
}

def load_real_delay_source(kind: str) -> pd.DataFrame:
    path = REAL_SOURCES.get(kind)
    if path is None:
        raise ValueError(f"Unknown real-source: {kind}")
    df = pd.read_csv(
        path,
        usecols=["reference_date", "report_date", "count", "delay"],
        dtype={"count": "float32", "delay": "int16"},
        parse_dates=["reference_date", "report_date"]
    )
    df = df[df["delay"] >= 0].copy()
    if kind == "uscdc":
        df = df[df["delay"] > 0].copy()
    return df

# Cache (simple in-memory) so we don't re-read on every callback
_real_cache = {}

def get_real_df(kind: str) -> pd.DataFrame:
    if kind not in _real_cache:
        _real_cache[kind] = load_real_delay_source(kind)
    return _real_cache[kind]

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

def safe_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12):
    den2 = np.where(np.abs(den) < eps, eps, den)
    return num / den2

def compute_psd(series: np.ndarray, dt: float = 1.0):
    """
    Return positive frequencies and power spectrum |FFT|^2 for a real series.
    """
    x = np.asarray(series, dtype=float)
    n = len(x)
    if n < 2:
        return np.array([]), np.array([])
    X = fft(x)
    freqs = fftfreq(n, d=dt)
    power = np.abs(X) ** 2
    pos = freqs > 0
    return freqs[pos], power[pos]

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

    dash_app.layout = html.Div([
        html.Div([

            # ===== Row 1 =====
            html.Div([
                dcc.Graph(id="result-plot", style={"height": "420px"})
            ], className="card area-main"),

            html.Div([
                dcc.Graph(id="influence-plot", style={"height": "400px"})
            ], className="card area-infl"),

            # ===== Row 2 =====
            html.Div([  # subgrid for three mini plots
                html.Div([dcc.Graph(id="psu-plot", style={"height": f"{MINI_H}px"})], className="mini-card"),
                html.Div([dcc.Graph(id="delay-plot", style={"height": f"{MINI_H}px"})], className="mini-card"),
                html.Div([dcc.Graph(id="psd-plot", style={"height": f"{MINI_H}px"})], className="mini-card"),
            ], className="area-minis minis-subgrid"),

            html.Div([
                dcc.Graph(id="attenuation-plot", style={"height": f"{MINI_H + 80}px"})
            ], className="card area-atten"),

            # ===== Row 3 =====
            html.Div([
                html.H4("Control Zone", style={"margin": "0 0 8px 0"}),

                html.Div([
                    html.Div([
                        html.Div([
                            html.Label("Select Region:"),
                            dcc.Dropdown(id="geo-dropdown",
                                         options=[{"label": g, "value": g} for g in geo_options],
                                         value=geo_options[0], clearable=False),

                            html.Div(style={"height": "10px"}),

                            html.Label("Amplitude of extra high-freq"),
                            dcc.Slider(id="hf-strength", min=0, max=3, step=0.1, value=0.5,
                                       tooltip={"placement": "bottom"},
                                       marks={0: "0", 1: "1", 2: "2", 3: "3"}, included=False),

                            html.Div(style={"height": "10px"}),

                            html.Label("Frequency (cycles/day) of extra high-freq"),
                            dcc.Slider(id="hf-freq", min=0.05, max=0.5, step=0.01, value=0.25,
                                       tooltip={"placement": "bottom"},
                                       marks={0.05: "0.05", 0.25: "0.25", 0.5: "0.5"}, included=False),
                        ], className="controls-col"),

                        html.Div([
                            html.Label("Delay Source:"),
                            dcc.Dropdown(
                                id="delay-source",
                                options=[
                                    {"label": "Synthetic (Gamma)", "value": "synthetic"},
                                    {"label": "Real (30 Provinces in China linelist)", "value": "cn30"},
                                    {"label": "Real (US CDC linelist)", "value": "uscdc"},
                                    {"label": "Real (Hong Kong linelist)", "value": "hk"},
                                ],
                                value="synthetic", clearable=False
                            ),
                            html.Div(id="real-source-info", style={"color": "blue", "marginTop": "5px"}),

                            html.Div([
                                html.Label("Gamma Mean:"),
                                dcc.Input(id="mean", type="number", value=5, step=0.1, style={"width": "100%"}),
                                html.Div(style={"height": "8px"}),
                                html.Label("Gamma Scale:"),
                                dcc.Input(id="scale", type="number", value=1, step=0.1, style={"width": "100%"}),
                            ], id="gamma-controls", style={"marginTop": "8px"}),

                            html.Div([
                                html.Label("Kernel Method:"),
                                dcc.RadioItems(
                                    id="kernel-method",
                                    options=[{"label": "Histogram", "value": "hist"},
                                             {"label": "KDE (weighted)", "value": "kde"}],
                                    value="hist",
                                    labelStyle={"display": "inline-block", "marginRight": "10px"}
                                ),
                                html.Div(style={"height": "8px"}),
                                html.Label("Window length (days, Y):"),
                                dcc.Input(id="window-days", type="number", value=30, min=1, step=1,
                                          style={"width": "100%"}),
                                html.Div(style={"height": "8px"}),
                                html.Label("Max delay to consider (days):"),
                                dcc.Input(id="max-delay", type="number", value=60, min=1, step=1,
                                          style={"width": "100%"}),
                                html.Div(style={"height": "8px"}),
                                html.Label("Window end date (t):"),
                                dcc.DatePickerSingle(id="window-end", display_format="YYYY-MM-DD",
                                                     placeholder="YYYY-MM-DD")
                            ], id="real-controls", style={"display": "none", "marginTop": "8px"}),
                        ], className="controls-col"),
                    ], className="controls-left-grid"),

                    html.Div([
                        html.Label("Influence threshold 𝑰_th"),
                        dcc.Slider(
                            id="influence-thresh",
                            min=0.0, max=1.0, step=0.01, value=0.10,
                            tooltip={"placement": "bottom"},
                            marks={0: "0.00", 0.1: "0.10", 0.25: "0.25", 0.5: "0.50", 0.75: "0.75", 1.0: "1.00"},
                            included=False
                        ),

                        html.Div(style={"height": "12px"}),
                        html.Button("Update", id="update-btn", n_clicks=0, style={"width": "100%"})
                    ], className="controls-right"),

                ], className="controls-outer-grid"),

            ], className="card area-controls"),

        ], className="grid-root")
    ], style={"padding": "12px"})

    # ---- Minimal CSS (inline style tags or external asset) ----
    dash_app.index_string = dash_app.index_string.replace(
        "</head>",
        """
        <style>
          /* ===== Top-level grid ===== */
          .grid-root {
            display: grid;
            grid-template-columns: 2fr 1fr;            
            grid-template-rows: auto auto auto;        
            grid-template-areas:
              "main infl"
              "minis atten"
              "controls controls";
            gap: 12px;
          }
          .area-main     { grid-area: main; }
          .area-infl     { grid-area: infl; }
          .area-minis    { grid-area: minis; }
          .area-atten    { grid-area: atten; }
          .area-controls { grid-area: controls; }

          /* minis */
          .minis-subgrid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
          }
          .card { background:#fff; padding:10px; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }
          .mini-card { background:#fff; padding:6px; border-radius:10px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }
          
          /* ===== Control Zone ===== */
          .controls-outer-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 16px;
            align-items: start; 
          }
          .controls-left-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
          }
          .controls-right {
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
          }
          .controls-grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
          .controls-col  { display:flex; flex-direction:column; }
        </style>
        </head>
        """
    )

    # ------- Callbacks  -------
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
            df_real = get_real_df(delay_source)
            total_count = df_real["count"].sum()
            min_date = df_real["reference_date"].min().date()
            max_date = df_real["reference_date"].max().date()
            return f"Total counts: {int(total_count)}, available window end date: {min_date} to {max_date}"
        except Exception as e:
            return f"Error loading real source: {e}"

    @dash_app.callback(
        Output("result-plot", "figure"),
        Output("psu-plot", "figure"),
        Output("delay-plot", "figure"),
        Output("psd-plot", "figure"),
        Output("influence-plot", "figure"),
        Output("attenuation-plot", "figure"),
        Input("update-btn", "n_clicks"),
        Input("geo-dropdown", "value"),
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
    def update_figures(n_clicks, _trigger_geo, delay_source, mean, scale,
                       kernel_method, window_days, max_delay, window_end_str,
                       hf_strength, hf_freq, I_thresh=0.1):
        geo_value = _trigger_geo
        mask = (df_full["geo_value"] == geo_value)
        df_geo = df_full.loc[mask, ["time_value", "JHU-Cases"]].sort_values("time_value")
        time = df_geo["time_value"].to_numpy(copy=False)
        confirmed = df_geo["JHU-Cases"].to_numpy(copy=False).astype(np.float32, copy=False)

        if delay_source == "synthetic":
            L = min(len(confirmed), (max_delay if max_delay else 60) + 1)
            delay_kernel = get_gamma_delay(L, mean, scale)
            label = f"Gamma(mean={mean}, scale={scale})"
            mu = np.sum(np.arange(L) * delay_kernel)
            var = np.sum((np.arange(L) ** 2) * delay_kernel) - mu**2
            gamma_fit_txt = f" (μ≈{mu:.2f}, σ²≈{var:.2f})"
        else:
            real_df = get_real_df(delay_source)
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

        # --- Delay spectrum (make available for mini-plot) ---
        delay_fft_complex = fft(delay_kernel)
        A = np.abs(delay_fft_complex)
        A2 = A ** 2  # |G(f)|^2 (power response of delay)
        freqs = fftfreq(len(delay_kernel), d=1.0)
        pos = freqs > 0  # positive frequencies only

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

        # ----- 1) MAIN TIME-DOMAIN FIG -----
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(
            x=time, y=confirmed, mode='lines', name='Confirmed Cases',
            line=dict(color=COLORS["confirmed"])
        ))

        # Wiener pair (solid symptomatic, dashed reconv)
        fig_main.add_trace(go.Scatter(
            x=time, y=wiener_curve, mode='lines',
            name='PsU-Wiener',
            line=dict(color=COLORS["wiener"])
        ))
        fig_main.add_trace(go.Scatter(
            x=time, y=wiener_reconv, mode='lines',
            name=f'PsD-Wiener (RMSE {rmse_wiener:.1f})',
            line=dict(color=COLORS["wiener"], dash='dash')
        ))

        # Alt pair
        fig_main.add_trace(go.Scatter(
            x=time, y=altered_infection, mode='lines',
            name='PsU-Wiener Alt (+hi-freq)',
            line=dict(color=COLORS["alt"])
        ))
        fig_main.add_trace(go.Scatter(
            x=time, y=altered_reconv, mode='lines',
            name=f'PsD-Wiener Alt (RMSE {rmse_altered:.1f})',
            line=dict(color=COLORS["alt"], dash='dash')
        ))

        # Tikhonov pair
        fig_main.add_trace(go.Scatter(
            x=time, y=tikhonov_curve, mode='lines',
            name=f'PsU-Tikhonov (λ={lambda_tikh:.3g})',
            line=dict(color=COLORS["tikhonov"])
        ))
        fig_main.add_trace(go.Scatter(
            x=time, y=tikhonov_reconv, mode='lines',
            name=f'PsD-Tikhonov (RMSE {rmse_tikh:.1f})',
            line=dict(color=COLORS["tikhonov"], dash='dash')
        ))
        fig_main.update_layout(title="Upstream vs Downstream: Time-Series Comparison",
                               xaxis_title="Date", yaxis_title="Cases", height=420,
                               legend=dict(orientation="h", yanchor="bottom", y=-0.7, xanchor="center", x=0.5),
                               margin=dict(t=100))

        # ----- 2) PSU (Upstream power spectrum of symptomatic curves) -----
        # Upstream = symptomatic reconstructions: wiener_curve, altered_infection, tikhonov_curve
        f_wu, P_wu = compute_psd(wiener_curve)
        f_au, P_au = compute_psd(altered_infection)
        f_tu, P_tu = compute_psd(tikhonov_curve)

        fig_psu = go.Figure()
        # shorter labels → less chance to wrap
        if f_wu.size: fig_psu.add_trace(go.Scatter(x=f_wu, y=P_wu, mode="lines",
                                                   name="PsU-Wiener", line=dict(color=COLORS["wiener"])))
        if f_au.size: fig_psu.add_trace(go.Scatter(x=f_au, y=P_au, mode="lines",
                                                   name="PsU-Wiener Alt", line=dict(color=COLORS["alt"])))
        if f_tu.size: fig_psu.add_trace(go.Scatter(x=f_tu, y=P_tu, mode="lines",
                                                   name="PsU-Tikhonov", line=dict(color=COLORS["tikhonov"])))

        fig_psu.update_layout(
            title="PsU(f) · Upstream Power Spectrum",
            xaxis_title="Frequency (cycles/day)", yaxis_title="Power",
            height=MINI_H, margin=MINI_MARGIN, legend=LEGEND
        )

        # ----- 3) Delay Spectrum (|G(f)|^2) with secondary axis for 100·(1 - |G(f)|^2) -----
        from plotly.subplots import make_subplots

        fig_delay = make_subplots(specs=[[{"secondary_y": True}]])

        if np.any(pos):
            fpos = freqs[pos]
            G2 = A2[pos]  # |G(f)|^2
            att = 100.0 * (1.0 - G2)  # 100·(1 - |G(f)|^2)

            # Left y-axis: power response |G(f)|^2
            fig_delay.add_trace(
                go.Scatter(x=fpos, y=G2, mode="lines",
                           name="|G(f)|² (delay power)",
                           line=dict(color=COLORS["delay"])),
                secondary_y=False
            )

            # Right y-axis: percent attenuation 100·(1 - |G(f)|^2)
            fig_delay.add_trace(
                go.Scatter(x=fpos, y=att, mode="lines",
                           name="100·(1 − |G(f)|²)",
                           line=dict(dash="dash")),
                secondary_y=True
            )

        fig_delay.update_layout(
            title="Delay Spectrum · |G(f)|²  &  100·(1 − |G(f)|²)",
            height=MINI_H, margin=MINI_MARGIN, legend=LEGEND
        )
        fig_delay.update_xaxes(title_text="Frequency (cycles/day)")
        fig_delay.update_yaxes(title_text="Power |G(f)|²", range=[0, 1], secondary_y=False)
        fig_delay.update_yaxes(title_text="Attenuation (%)", range=[0, 100], secondary_y=True)


        # ----- 4) PSD (Downstream power spectrum of confirmed + reconv curves) -----
        # Downstream = confirmed cases & reconvolved curves
        f_c, P_c = compute_psd(confirmed)
        f_wd, P_wd = compute_psd(wiener_reconv)
        f_ad, P_ad = compute_psd(altered_reconv)
        f_td, P_td = compute_psd(tikhonov_reconv)

        fig_psd = go.Figure()
        if f_c.size:
            fig_psd.add_trace(go.Scatter(x=f_c, y=P_c, mode="lines",
                                         name="Confirmed (obs.)",
                                         line=dict(color=COLORS["confirmed"])))
        if f_wd.size:
            fig_psd.add_trace(go.Scatter(x=f_wd, y=P_wd, mode="lines",
                                         name="PsD-Wiener", line=dict(color=COLORS["wiener"])))
        if f_ad.size:
            fig_psd.add_trace(go.Scatter(x=f_ad, y=P_ad, mode="lines",
                                         name="PsD-Wiener Alt", line=dict(color=COLORS["alt"])))
        if f_td.size:
            fig_psd.add_trace(go.Scatter(x=f_td, y=P_td, mode="lines",
                                         name="PsD-Tikhonov", line=dict(color=COLORS["tikhonov"])))

        fig_psd.update_layout(
            title="PsD(f) · Downstream Power Spectrum",
            xaxis_title="Frequency (cycles/day)", yaxis_title="Power",
            height=MINI_H, margin=MINI_MARGIN, legend=LEGEND
        )

        # ----- 5) Influence (right tall figure) -----
        # Split your old fig_fft into a standalone "influence" figure;
        # keep the dual-y: power of delay vs I(f)/S(f) curves + f_cut markers.

        fig_infl = make_subplots(specs=[[{"secondary_y": True}]])
        fig_infl.add_trace(go.Scatter(x=freqs[pos], y=A2[pos], mode='lines', line=dict(color=COLORS["delay"]),
                                      name="Delay power spectrum"),
                           secondary_y=False)
        fig_infl.add_trace(go.Scatter(x=freqs[pos], y=influence_base[pos], mode='lines',
                                      line=dict(color=COLORS["wiener"]),
                                      name="I(f)-Wiener"),
                           secondary_y=True)
        fig_infl.add_trace(go.Scatter(x=freqs[pos], y=influence_alt[pos], mode='lines',
                                      line=dict(color=COLORS["alt"]),
                                      name="I(f)-Wiener altered"),
                           secondary_y=True)
        fig_infl.add_trace(go.Scatter(x=freqs[pos], y=H_tikh[pos], mode='lines',
                                      line=dict(color=COLORS["tikhonov"]),
                                      name="S(f)-Tikhonov"),
                           secondary_y=True)

        def add_cut_line(fig, f_val, color, label):
            if f_val is None or not np.isfinite(f_val): return
            fig.add_trace(go.Scatter(x=[f_val, f_val], y=[0, 1], mode="lines",
                                     line=dict(color=color, dash="dash", width=2),
                                     name=label, yaxis="y2"), secondary_y=True)

        add_cut_line(fig_infl, f_cut_base, "red", f"f_cut(Wiener)≈{f_cut_base:.3f} (P≈{P_cut_base:.1f}d)")
        add_cut_line(fig_infl, f_cut_alt, "orange", f"f_cut(Wiener alt)≈{f_cut_alt:.3f} (P≈{P_cut_alt:.1f}d)")
        add_cut_line(fig_infl, f_cut_tikh, "green", f"f_cut(Tikh)≈{f_cut_tikh:.3f} (P≈{P_cut_tikh:.1f}d)")

        fig_infl.update_layout(
            title="Delay Spectrum & SNR-aware Influence",
            height=420,
            legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
            margin=dict(t=70, b=100, l=60, r=10)
        )
        fig_infl.add_annotation(
            x=0.5, y=1.12, xref="paper", yref="paper",
            text=f"λ_base≈{lambda_base:.3g}, λ_alt≈{lambda_alt:.3g}, λ_tikh≈{lambda_tikh2:.3g}",
            showarrow=False, xanchor="center", font=dict(size=14)
        )
        fig_infl.update_xaxes(title_text="Frequency (cycles/day)")
        fig_infl.update_yaxes(title_text="Power |G(f)|²", range=[0, 1], secondary_y=False)
        fig_infl.update_yaxes(title_text="Influence I(f)", range=[0, 1], secondary_y=True)

        # ----- Att Fig -----
        fig_att = go.Figure()

        # Only plot where we have both upstream and downstream PSDs
        if f_wu.size and f_wd.size:
            # assume same freq grid; if not, you can interpolate one to the other
            pct_w = 100.0 * (1.0 - safe_ratio(P_wd, P_wu))
            fig_att.add_trace(go.Scatter(x=f_wu, y=pct_w, mode="lines",
                                         name="Wiener",
                                         line=dict(color=COLORS["wiener"])))

        if f_au.size and f_ad.size:
            pct_a = 100.0 * (1.0 - safe_ratio(P_ad, P_au))
            fig_att.add_trace(go.Scatter(x=f_au, y=pct_a, mode="lines",
                                         name="Wiener Alt",
                                         line=dict(color=COLORS["alt"])))

        if f_tu.size and f_td.size:
            pct_t = 100.0 * (1.0 - safe_ratio(P_td, P_tu))
            fig_att.add_trace(go.Scatter(x=f_tu, y=pct_t, mode="lines",
                                         name="Tikhonov",
                                         line=dict(color=COLORS["tikhonov"])))

        fig_att.update_layout(
            title="Percent Attenuation vs Frequency  -  100·(1 − PsD/PsU)",
            xaxis_title="Frequency (cycles/day)",
            yaxis_title="Attenuation (%)",
            height=360,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
            margin=dict(t=60, b=60, l=60, r=10)
        )


        # ----- Return all five figures -----
        return fig_main, fig_psu, fig_delay, fig_psd, fig_infl, fig_att

    return dash_app