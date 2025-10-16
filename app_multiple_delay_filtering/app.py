import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, gaussian_kde
from numpy.fft import rfft, rfftfreq
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

# =============== Load Base Data (for timeline) ===============
df_full = pd.read_csv("./data/combined_state_no_revision.csv")
df_full["time_value"] = pd.to_datetime(df_full["time_value"])
df_full = df_full.sort_values("time_value")
geo_options = sorted(df_full["geo_value"].unique())

# =============== Real delay sources (for Symptom→Case only) ===============
REAL_SOURCES = {
    "cn30": "./data/cn30_linelist_reporting_delay_from_symptom_onset.csv",
    "uscdc": "./data/uscdc_linelist_reporting_delay_from_symptom_onset.csv",
    "hk": "./data/hk_linelist_reporting_delay_from_symptom_onset.csv",
}


def load_real_delay_source(kind: str) -> pd.DataFrame:
    path = REAL_SOURCES.get(kind)
    if path is None:
        raise ValueError(f"Unknown real source: {kind}")
    df = pd.read_csv(path)
    # Expected columns: reference_date, delay, count
    df["reference_date"] = pd.to_datetime(df["reference_date"])  # windowing anchor
    df["count"] = df["count"].astype(float)
    df["delay"] = df["delay"].astype(int)
    df = df[df["delay"] >= 0].copy()
    if kind == "uscdc":
        # CDC dataset often includes 0 as clerical same-day; we commonly drop to avoid spike.
        df = df[df["delay"] > 0].copy()
    return df


_real_cache: Dict[str, pd.DataFrame] = {}

def get_real_df(kind: str) -> pd.DataFrame:
    if kind not in _real_cache:
        _real_cache[kind] = load_real_delay_source(kind)
    return _real_cache[kind]


# =============== Helpers =============================

def normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    s = arr.sum()
    if s > 0:
        return arr / s
    return arr


def get_gamma_delay(length: int, mean: float, scale: float) -> np.ndarray:
    """Gamma delay with shape a = mean/scale; support [0, length-1]."""
    x = np.arange(length)
    a = max(mean / max(scale, 1e-9), 1e-6)
    delay = gamma_dist.pdf(x, a=a, scale=max(scale, 1e-9))
    return normalize(delay)


def build_delay_kernel_for_window(
    df_delays: pd.DataFrame,
    window_end: pd.Timestamp,
    window_days: int = 30,
    max_delay: int = 60,
    method: str = "hist",
) -> Tuple[np.ndarray, int]:
    """Aggregate a time-windowed delay kernel from line list counts.

    Returns (kernel, n_obs_in_window).
    """
    t0 = window_end - pd.Timedelta(days=window_days)
    window = df_delays[(df_delays["reference_date"] > t0) & (df_delays["reference_date"] <= window_end)]
    if window.empty:
        return np.zeros(max_delay + 1, dtype=float), 0

    agg = window.groupby("delay", as_index=False)["count"].sum().sort_values("delay")
    agg = agg[agg["delay"] <= max_delay]
    grid = np.arange(0, max_delay + 1)

    if method == "hist":
        hist = np.zeros_like(grid, dtype=float)
        delays = agg["delay"].to_numpy()
        counts = agg["count"].to_numpy()
        hist[delays] = counts
        return normalize(hist), int(counts.sum())

    # KDE (non-parametric smoothing)
    delays = agg["delay"].to_numpy()
    counts = agg["count"].to_numpy()
    if len(delays) < 2 or counts.sum() <= 0:
        # Fallback to histogram if too few points
        hist = np.zeros_like(grid, dtype=float)
        hist[delays] = counts
        return normalize(hist), int(counts.sum())

    kde = gaussian_kde(delays, weights=counts)
    pdf = np.clip(kde(grid), 0, None)
    return normalize(pdf), int(counts.sum())


def cascade_forward(infections: np.ndarray, kernels: Dict[str, np.ndarray], T: int):
    """Forward cascade through the stages, cropped to horizon T."""
    conv = lambda x, k: np.convolve(x, k, mode="full")[:T]
    S = conv(infections, kernels['inf2sympt'])
    C = conv(S,          kernels['sympt2case'])
    H = conv(S,          kernels['sympt2hosp'])
    D = conv(H,          kernels['hosp2death'])
    Drep = conv(D,       kernels['death2report'])
    return S, C, H, D, Drep


def composed_kernel(kernels: Dict[str, np.ndarray]) -> np.ndarray:
    """Overall kernel from infections to reported deaths."""
    k = kernels['inf2sympt']
    for name in ['sympt2case', 'sympt2hosp', 'hosp2death', 'death2report']:
        k = np.convolve(k, kernels[name], mode='full')
    return normalize(k)


def toy_infections(T: int) -> np.ndarray:
    """A slightly richer synthetic infection curve (multi-modal)."""
    y = np.zeros(T)
    # smooth pulses
    def add_bump(center, width, height):
        x = np.arange(T)
        yb = height * np.exp(-0.5 * ((x - center) / max(width, 1)) ** 2)
        return yb

    y += add_bump(30, 5, 120)
    y += add_bump(75, 7, 80)
    y += add_bump(130, 3, 160)
    # mild plateau
    if T > 190:
        y[160:190] += 40
    return y[:T]


def kernel_power_spectrum(k: np.ndarray, d: float = 1.0):
    """Return (freqs, power) using rFFT; freq in cycles per unit d (day)."""
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size == 0:
        return np.array([]), np.array([])
    P = np.abs(rfft(k)) ** 2
    f = rfftfreq(k.size, d=d)
    # Drop the zero frequency for plotting clarity (DC spike)
    mask = f > 0
    return f[mask], P[mask]


def freq_to_period(freqs: np.ndarray) -> np.ndarray:
    # Avoid divide-by-zero; only call on f>0
    return 1.0 / freqs


# =============== Dash App UI =============================================
app = Dash(__name__)
app.title = "COVID Multi-Stage Delay Cascade"

controls_col = html.Div(
    [
        html.H2("Impact from Delay Distributions", style={"marginTop": 0}),
        html.Label("Region (for timeline):"),
        dcc.Dropdown(
            id="geo-dropdown",
            options=[{"label": g, "value": g} for g in geo_options],
            value=geo_options[0],
            clearable=False,
        ),
        html.Hr(),

        # Stage 1: Infection → Symptom
        html.H4("Infection → Symptom (Gamma)"),
        html.Div(
            [
                html.Span("Mean"), dcc.Input(id="inf2sympt-mean", type="number", value=5, step=0.5, style={"width": 90}),
                html.Span("Scale", style={"marginLeft": 10}), dcc.Input(id="inf2sympt-scale", type="number", value=1, step=0.1, style={"width": 90}),
            ],
            style={"display": "flex", "gap": "8px", "alignItems": "center"},
        ),
        html.Br(),

        # Stage 2: Symptom → Case
        html.H4("Symptom → Case Reporting"),
        dcc.Dropdown(
            id="sympt2case-source",
            options=[
                {"label": "Synthetic (Gamma)", "value": "synthetic"},
                {"label": "Real (30 Provinces China)", "value": "cn30"},
                {"label": "Real (US CDC)", "value": "uscdc"},
                {"label": "Real (Hong Kong)", "value": "hk"},
            ],
            value="synthetic",
            clearable=False,
        ),
        html.Div(
            id="sympt2case-controls",
            children=[
                html.Div([
                    html.Span("Mean"), dcc.Input(id="sympt2case-mean", type="number", value=3, step=0.5, style={"width": 90}),
                    html.Span("Scale", style={"marginLeft": 10}), dcc.Input(id="sympt2case-scale", type="number", value=1, step=0.1, style={"width": 90}),
                ], style={"display": "flex", "gap": "8px", "alignItems": "center"}),
            ],
            style={"marginTop": "6px"},
        ),
        html.Div(
            id="real-source-controls",
            children=[
                html.Div([
                    html.Span("Real-source method:"),
                    dcc.RadioItems(
                        id="real-method",
                        options=[{"label": "Histogram", "value": "hist"}, {"label": "KDE", "value": "kde"}],
                        value="hist",
                        inline=True,
                    ),
                ])
            ],
            style={"display": "none", "marginTop": "6px"},
        ),
        html.Br(),

        # Stage 3: Symptom → Hosp
        html.H4("Symptom → Hospitalization (Gamma)"),
        html.Div(
            [
                html.Span("Mean"), dcc.Input(id="sympt2hosp-mean", type="number", value=7, step=0.5, style={"width": 90}),
                html.Span("Scale", style={"marginLeft": 10}), dcc.Input(id="sympt2hosp-scale", type="number", value=1, step=0.1, style={"width": 90}),
            ],
            style={"display": "flex", "gap": "8px", "alignItems": "center"},
        ),
        html.Br(),

        # Stage 4: Hosp → Death
        html.H4("Hospitalization → Death (Gamma)"),
        html.Div(
            [
                html.Span("Mean"), dcc.Input(id="hosp2death-mean", type="number", value=10, step=0.5, style={"width": 90}),
                html.Span("Scale", style={"marginLeft": 10}), dcc.Input(id="hosp2death-scale", type="number", value=1, step=0.1, style={"width": 90}),
            ],
            style={"display": "flex", "gap": "8px", "alignItems": "center"},
        ),
        html.Br(),

        # Stage 5: Death → Report
        html.H4("Death → Report (Gamma)"),
        html.Div(
            [
                html.Span("Mean"), dcc.Input(id="death2report-mean", type="number", value=3, step=0.5, style={"width": 90}),
                html.Span("Scale", style={"marginLeft": 10}), dcc.Input(id="death2report-scale", type="number", value=1, step=0.1, style={"width": 90}),
            ],
            style={"display": "flex", "gap": "8px", "alignItems": "center"},
        ),
        html.Hr(),

        html.H4("Spectrum Options"),
        html.Div([
            dcc.RadioItems(
                id="fft-x-mode",
                options=[{"label": "Frequency (cycles/day)", "value": "freq"}, {"label": "Period (days)", "value": "period"}],
                value="period",
                inline=False,
            ),
            dcc.Checklist(
                id="fft-logy",
                options=[{"label": "Log scale (power)", "value": "logy"}],
                value=["logy"],
            ),
        ]),

        html.Button("Update", id="update-btn", n_clicks=0, style={"marginTop": 12}),
    ],
    style={
        "width": "28%",
        "minWidth": "320px",
        "maxWidth": "420px",
        "padding": "16px",
        "borderRight": "1px solid #eee",
        "height": "100vh",
        "overflowY": "auto",
        "boxSizing": "border-box",
    },
)

# --- Figures area: fig1 top, fig2+fig3 side-by-side
figs_col = html.Div(
    [
        dcc.Graph(id="result-plot", style={"height": "38vh"}),
        html.Div(
            [
                html.Div([dcc.Graph(id="kernel-plot", style={"height": "38vh"})], style={"width": "49%"}),
                html.Div([dcc.Graph(id="fft-plot", style={"height": "38vh"})], style={"width": "49%"}),
            ],
            style={"display": "flex", "gap": "2%", "alignItems": "stretch"},
        ),
    ],
    style={
        "width": "72%",
        "padding": "12px 16px",
        "height": "100vh",
        "overflowY": "auto",
        "boxSizing": "border-box",
    },
)

app.layout = html.Div(
    [controls_col, figs_col],
    style={"display": "flex", "flexDirection": "row", "width": "100vw", "height": "100vh", "margin": 0},
)


# =============== Callbacks ================================================
@app.callback(
    Output("sympt2case-controls", "style"),
    Output("real-source-controls", "style"),
    Input("sympt2case-source", "value"),
)
def toggle_sympt2case_controls(source):
    if source == "synthetic":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


@app.callback(
    Output("result-plot", "figure"),
    Output("kernel-plot", "figure"),
    Output("fft-plot", "figure"),
    Input("update-btn", "n_clicks"),
    State("geo-dropdown", "value"),
    State("inf2sympt-mean", "value"), State("inf2sympt-scale", "value"),
    State("sympt2case-source", "value"),
    State("sympt2case-mean", "value"), State("sympt2case-scale", "value"),
    State("real-method", "value"),
    State("sympt2hosp-mean", "value"), State("sympt2hosp-scale", "value"),
    State("hosp2death-mean", "value"), State("hosp2death-scale", "value"),
    State("death2report-mean", "value"), State("death2report-scale", "value"),
    State("fft-x-mode", "value"), State("fft-logy", "value"),
)

def update_figures(
    n_clicks,
    geo_value,
    m1, s1,
    sympt2case_source,
    m2, s2,
    real_method,
    m3, s3, m4, s4, m5, s5,
    fft_x_mode, fft_logy_vals,
):
    # --- Timeline for selected region
    df = df_full[df_full["geo_value"] == geo_value]
    T = len(df)
    time = df["time_value"].values
    last_date = pd.to_datetime(time[-1]) if T > 0 else pd.Timestamp.today().normalize()

    # --- Example infections
    infections = toy_infections(max(T, 220))[:T] if T > 0 else np.zeros(200)

    # --- Build kernels
    kernels: Dict[str, np.ndarray] = {}
    kernels['inf2sympt'] = get_gamma_delay(60, float(m1), float(s1))

    if sympt2case_source == "synthetic":
        kernels['sympt2case'] = get_gamma_delay(60, float(m2), float(s2))
    else:
        real_df = get_real_df(sympt2case_source)
        # Align window end to timeline end; cap within available reference_date range
        real_end = min(last_date, real_df["reference_date"].max())
        k_real, n_obs = build_delay_kernel_for_window(real_df, real_end, window_days=30, max_delay=60, method=real_method)
        if n_obs == 0 or k_real.sum() == 0:
            # Fallback gently to synthetic default to avoid empty curves
            k_real = get_gamma_delay(60, 3.0, 1.0)
        kernels['sympt2case'] = k_real

    kernels['sympt2hosp'] = get_gamma_delay(60, float(m3), float(s3))
    kernels['hosp2death'] = get_gamma_delay(60, float(m4), float(s4))
    kernels['death2report'] = get_gamma_delay(60, float(m5), float(s5))

    # --- Cascade forward
    S, C, H, D, Drep = cascade_forward(infections, kernels, T)

    # --- Main plot (ensure xticks/xlabel visible)
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=time, y=infections, name="Infections"))
    fig_main.add_trace(go.Scatter(x=time, y=S, name="Symptomatics"))
    fig_main.add_trace(go.Scatter(x=time, y=C, name="Cases"))
    fig_main.add_trace(go.Scatter(x=time, y=H, name="Hospitalizations"))
    fig_main.add_trace(go.Scatter(x=time, y=D, name="Deaths"))
    fig_main.add_trace(go.Scatter(x=time, y=Drep, name="Reported Deaths"))
    fig_main.update_layout(
        title=f"Cascade Simulation — {geo_value.upper()}",
        xaxis_title="Date",
        yaxis_title="Counts",
        xaxis=dict(showticklabels=True),
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(l=30, r=10, t=50, b=40),
        height=420,
    )

    # --- Kernel plot (include composed kernel) with legend below
    fig_kernel = go.Figure()
    for name, k in kernels.items():
        fig_kernel.add_trace(go.Scatter(y=k, mode="lines", name=name))
    K_comp = composed_kernel(kernels)
    fig_kernel.add_trace(go.Scatter(y=K_comp, mode="lines", name="composed: inf→reported"))
    fig_kernel.update_layout(
        title="Delay Kernels (normalized)",
        xaxis_title="Delay (days)", yaxis_title="Probability",
        legend=dict(orientation="h", y=-0.25, x=0),  # below the plot
        margin=dict(l=30, r=10, t=50, b=80),
    )

    # --- FFT plot (rFFT, optional period axis and log-y) with legend below
    fig_fft = go.Figure()
    for name, k in {**kernels, "composed": K_comp}.items():
        f, P = kernel_power_spectrum(k)
        if f.size == 0:
            continue
        if fft_x_mode == "period":
            x = freq_to_period(f)
            x_title = "Period (days per cycle)"
        else:
            x = f
            x_title = "Frequency (cycles/day)"
        fig_fft.add_trace(go.Scatter(x=x, y=P, mode="lines", name=name))

    yaxis_type = "log" if (fft_logy_vals and "logy" in fft_logy_vals) else "linear"
    fig_fft.update_layout(
        title="Kernel Power Spectrum (rFFT)",
        xaxis_title=x_title,
        yaxis_title="Power",
        yaxis_type=yaxis_type,
        legend=dict(orientation="h", y=-0.25, x=0),  # below the plot
        margin=dict(l=30, r=10, t=50, b=80),
    )

    return fig_main, fig_kernel, fig_fft


if __name__ == "__main__":
    app.run(debug=False)
