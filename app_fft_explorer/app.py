# app_fft_explorer/app.py  — memory friendly factory version
import base64
import io
import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from scipy.fft import rfft, irfft, rfftfreq
from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_PATH = "./data/ar_state.csv"

# ------------ simple in-process cache ------------
_SERVER_CACHE = {}  # key -> pd.DataFrame

def _optimize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast dtypes to cut memory; make geo categorical; keep time as datetime64."""
    df = df.copy()
    if "time_value" in df.columns:
        df["time_value"] = pd.to_datetime(df["time_value"], errors="coerce")
    if "geo_value" in df.columns and df["geo_value"].dtype != "category":
        df["geo_value"] = df["geo_value"].astype("category")
    for c in df.columns:
        if c in ("time_value", "geo_value"):
            continue
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    return df

def _cache_key(tag: str, df: pd.DataFrame) -> str:
    h = hashlib.md5()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    return f"fft:{tag}:{h.hexdigest()}"

def cache_put_df(df: pd.DataFrame, tag: str="uploaded") -> str:
    key = _cache_key(tag, df)
    _SERVER_CACHE[key] = df
    return key

def cache_get_df(key: str) -> Optional[pd.DataFrame]:
    return _SERVER_CACHE.get(key)

# ============== Data Loading ==============
def load_default_df() -> pd.DataFrame:
    df = pd.read_csv(DEFAULT_PATH)
    df = _optimize_df(df)
    return df

def parse_uploaded_contents(contents: str, filename: str) -> pd.DataFrame:
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
    df = _optimize_df(df)
    return df

def detect_schema(df: pd.DataFrame):
    # prefer wide format with geo/time + numeric columns
    if ("geo_value" in df.columns) and ("time_value" in df.columns):
        non_meta = [c for c in df.columns if c not in ["geo_value", "time_value"]]
        numeric_signals = [c for c in non_meta if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_signals) >= 1:
            out = df.copy()
            out["time_value"] = pd.to_datetime(out["time_value"], errors="coerce")
            states = sorted(out["geo_value"].dropna().unique().tolist())
            signals = sorted(numeric_signals)
            return {"mode": "wide", "states": states, "signals": signals, "df": out}

    # fallback: simple 2-col value
    time_col = None
    for cand in ["time", "date", "time_value"]:
        if cand in df.columns:
            time_col = cand
            break
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    simple_df = pd.DataFrame()
    if time_col and len(numeric_cols) >= 1:
        signal_col = numeric_cols[0]
        simple_df["time_value"] = pd.to_datetime(df[time_col], errors="coerce")
        simple_df["geo_value"] = "uploaded"
        simple_df["value"] = pd.to_numeric(df[signal_col], errors="coerce")
    elif not time_col and len(numeric_cols) >= 1:
        signal_col = numeric_cols[0]
        n = len(df)
        base = pd.Timestamp("2000-01-01")
        simple_df["time_value"] = base + pd.to_timedelta(np.arange(n), unit='D')
        simple_df["geo_value"] = "uploaded"
        simple_df["value"] = pd.to_numeric(df[signal_col], errors="coerce")
    else:
        raise ValueError("Could not detect a usable schema.")
    simple_df = _optimize_df(simple_df)
    return {"mode": "simple", "states": ["uploaded"], "signals": ["value"], "df": simple_df}

# ============== Simulated Data Factory (time series) ==============
def _rng(seed: Optional[int]):
    # Helper to create a numpy Generator with an optional seed
    return np.random.default_rng(None if seed in (None, "", "None") else int(seed))

def synth_white_gaussian(n: int, mean: float, std: float, seed: Optional[int]):
    """y[t] = mean + N(0, std^2)"""
    r = _rng(seed)
    y = mean + r.normal(0.0, std, size=n)
    return y

def synth_ar1_gaussian(n: int, phi: float, std: float, mean: float, seed: Optional[int]):
    """AR(1): y[t] = mean + phi*(y[t-1]-mean) + eps, eps~N(0, std^2)"""
    r = _rng(seed)
    y = np.zeros(n, dtype=float)
    y[0] = mean + r.normal(0.0, std)
    for t in range(1, n):
        y[t] = mean + phi * (y[t-1] - mean) + r.normal(0.0, std)
    return y

def synth_poisson(n: int, lam: float, seed: Optional[int]):
    """y[t] ~ Poisson(lam), returned as float"""
    r = _rng(seed)
    y = r.poisson(lam, size=n).astype(float)
    return y

def synth_piecewise_constant(n: int, n_segments: int, low: float, high: float,
                             noise_std: float, seed: Optional[int]):
    """Piecewise-constant mean with Gaussian noise."""
    r = _rng(seed)
    # Choose breakpoints and levels
    n_segments = max(1, int(n_segments))
    if n_segments > n:
        n_segments = n
    # Random break positions (sorted, unique), ensure start and end included
    if n_segments == 1:
        breaks = [0, n]
    else:
        cut_points = sorted(r.choice(np.arange(1, n), size=n_segments - 1, replace=False).tolist())
        breaks = [0] + cut_points + [n]
    levels = r.uniform(low, high, size=len(breaks) - 1)
    y = np.empty(n, dtype=float)
    for i in range(len(levels)):
        y[breaks[i]:breaks[i+1]] = levels[i]
    # Add small Gaussian noise
    if noise_std > 0:
        y = y + r.normal(0.0, noise_std, size=n)
    return y

def synth_spikes_on_constant(n: int, baseline: float, n_spikes: int,
                             amp_low: float, amp_high: float, width_min: int, width_max: int,
                             seed: Optional[int]):
    """Constant baseline plus several positive spikes (box/carrier) with random positions and widths."""
    r = _rng(seed)
    y = np.full(n, float(baseline))
    n_spikes = max(0, int(n_spikes))
    for _ in range(n_spikes):
        w = int(r.integers(width_min, width_max + 1))
        pos = int(r.integers(0, max(1, n - w)))
        amp = float(r.uniform(amp_low, amp_high))
        y[pos:pos + w] += amp
    return y

def make_simulated_df(kind: str, length: int, seed: Optional[int], params: dict) -> pd.DataFrame:
    """
    Build a DataFrame with columns [time_value, geo_value, value] for simulated data.
    geo_value is set to 'simulated'. time_value is daily from 2000-01-01.
    """
    length = max(5, int(length))
    base = pd.Timestamp("2000-01-01")
    if kind == "white":
        y = synth_white_gaussian(length, params.get("mean", 0.0), params.get("std", 1.0), seed)
    elif kind == "ar1":
        y = synth_ar1_gaussian(length, params.get("phi", 0.8), params.get("std", 1.0), params.get("mean", 0.0), seed)
    elif kind == "poisson":
        y = synth_poisson(length, params.get("lam", 10.0), seed)
    elif kind == "piecewise":
        y = synth_piecewise_constant(length, params.get("n_segments", 4),
                                     params.get("low", 0.0), params.get("high", 10.0),
                                     params.get("noise_std", 0.5), seed)
    elif kind == "spikes":
        y = synth_spikes_on_constant(length, params.get("baseline", 0.0),
                                     params.get("n_spikes", 5),
                                     params.get("amp_low", 5.0), params.get("amp_high", 20.0),
                                     params.get("width_min", 1), params.get("width_max", 5),
                                     seed)
    else:
        raise ValueError(f"Unknown simulated kind: {kind}")

    df = pd.DataFrame({
        "time_value": base + pd.to_timedelta(np.arange(length), unit="D"),
        "geo_value": "simulated",
        "value": y.astype(float)
    })
    df = _optimize_df(df)
    return df

# ============== FFT Utils (rFFT) ==============
def extract_signal(df, state, signal, date_cutoff="2020-12-31"):
    mask = (df["geo_value"] == state)
    if date_cutoff:
        cutoff = pd.to_datetime(date_cutoff)
        mask &= (df["time_value"] <= cutoff)
    cols = ["time_value", signal]
    curve = df.loc[mask, cols].dropna().sort_values("time_value")
    if curve.empty:
        return None, None, None
    t_raw = pd.to_datetime(curve["time_value"])
    y_raw = curve[signal].astype("float32").to_numpy(copy=False)
    y_raw = np.nan_to_num(y_raw, nan=0.0).astype("float32", copy=False)
    y_mean = float(np.mean(y_raw)) if len(y_raw) else 0.0
    return y_raw, t_raw, y_mean

def _truncate_and_decimate(t: pd.Series, y: np.ndarray, max_points=5000):
    """Keep last max_points and decimate further by stride to ~2000 points for plotting."""
    if len(y) > max_points:
        t = t.iloc[-max_points:]
        y = y[-max_points:]
    # pixel-aware simple decimation
    target = 2000
    if len(y) > target:
        stride = int(np.ceil(len(y) / target))
        t = t.iloc[::stride]
        y = y[::stride]
    return t, y

def preprocess_signal(y, t_raw, pad_length, pad_side="both"):
    if pad_side == "left":
        y_padded = np.concatenate([np.zeros(pad_length, dtype="float32"), y])
    else:
        y_padded = np.concatenate([np.zeros(pad_length, dtype="float32"), y, np.zeros(pad_length, dtype="float32")])
    full_time = pd.date_range(
        start=t_raw.iloc[0] - pd.to_timedelta(pad_length, unit='D'),
        periods=len(y_padded), freq='D'
    )
    return y_padded.astype("float32", copy=False), full_time

def pad_for_plot(y, pad_length, pad_side="both"):
    left = np.zeros(pad_length, dtype="float32")
    right = np.zeros(pad_length if pad_side == "both" else 0, dtype="float32")
    return np.concatenate([left, y.astype("float32", copy=False), right])

def compute_fft(y_padded, dt=1.0):
    N = len(y_padded)
    freqs = rfftfreq(N, d=dt)
    fft_vals = rfft(y_padded)
    return freqs, fft_vals

def apply_frequency_filter(freqs, fft_vals, low_cutoff, high_cutoff, method="hard"):
    # freqs >= 0
    fft_filtered = fft_vals.copy()
    if method == "gaussian":
        eps = 1e-12 if high_cutoff == 0 else 0.0
        weights = np.exp(-((freqs) / (high_cutoff + eps)) ** 2)
        fft_filtered *= weights
    else:
        band = (freqs >= low_cutoff) & (freqs <= high_cutoff)
        fft_filtered[~band] = 0
    return fft_filtered

def compute_power_spectrum(freqs, fft_vals):
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    return pos_freqs, power

def compute_energy_distribution(freqs_full, fft_vals_full):
    spectrum_full = np.abs(fft_vals_full) ** 2
    pos_mask = freqs_full > 0
    freqs_pos = freqs_full[pos_mask]
    P_pos = spectrum_full[pos_mask]
    if len(freqs_pos) == 0:
        return np.array([]), np.array([]), np.array([]), 1.0
    order = np.argsort(freqs_pos)
    f_sorted = freqs_pos[order]
    P_sorted = P_pos[order]
    total = float(np.sum(P_sorted))
    if total <= 0:
        pmf = np.zeros_like(P_sorted)
        cdf = np.zeros_like(P_sorted)
    else:
        pmf = P_sorted / total
        cdf = np.cumsum(pmf)
    df = float(np.mean(np.diff(f_sorted))) if len(f_sorted) > 1 else 1.0
    return f_sorted, pmf, cdf, df

def entropy_from_pdf(pdf_vals, mask=None):
    if pdf_vals is None or len(pdf_vals) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p = np.asarray(pdf_vals, dtype=float)
    p_safe = np.where(p > 0, p, 1e-300)
    H_total = float(-(p_safe * np.log(p_safe)).sum())
    K = (p > 0).sum()
    H_norm = float(H_total / np.log(K)) if K > 1 else 0.0
    if mask is not None and mask.any():
        p_band = p_safe[mask]
        H_band = float(-(p_band * np.log(p_band)).sum())
        band_share = H_band / H_total if H_total > 0 else 0.0
    else:
        H_band, band_share = 0.0, 0.0
    return H_total, H_band, H_norm, band_share

def bin_energy_pdf(pdf_freqs, pdf_vals, nbins=50, use_log=False):
    if pdf_freqs is None or len(pdf_freqs) == 0:
        return np.array([]), np.array([]), np.array([])
    f = np.asarray(pdf_freqs)
    p = np.asarray(pdf_vals)
    mask = f > 0
    f = f[mask]; p = p[mask]
    if len(f) == 0:
        return np.array([]), np.array([]), np.array([])
    if use_log:
        fmin, fmax = np.min(f), np.max(f)
        edges = np.logspace(np.log10(fmin), np.log10(fmax), nbins + 1)
    else:
        edges = np.linspace(np.min(f), np.max(f), nbins + 1)
    pdf_binned = np.zeros(nbins, dtype=float)
    idx = np.searchsorted(edges, f, side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    for k in range(nbins):
        pdf_binned[k] = p[idx == k].sum()
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, pdf_binned, edges

def create_fft_plot(
    full_time, y_plot_original, y_plot_recon,
    x_vals_all, power_all,            # full-spectrum x and power (light-green background)
    x_vals_display, power_display,    # filtered spectrum (kept for potential future use)
    band_mask,                        # boolean mask on x_vals_all
    xaxis_label,                      # "Frequency (cycles/day)" or "Period (days)"
    signal, state, yaxis_type="linear",
    frac_info=None,
):
    """
    Render two subplots:
      Row 1: Time-domain (original vs reconstructed), with an internal top-right legend.
      Row 2: Power spectrum (full background in light green; selected band overlay in dark green),
             with an internal top-right legend. X-axis can be frequency or period.
    Notes:
      - Global legend is disabled; we draw internal legends via shapes + annotations.
      - All comments are in English as requested.
    """

    # ===== Internal top-right legends (unchanged) =====
    def add_internal_legend_topright(fig, xdomain, ydomain, items,
                                     box_pad=0.010, row_gap=0.075,
                                     swatch_w=0.020, swatch_h=0.040,
                                     font_size=12, with_bg=True):
        # Draw compact legend inside the plot area
        x1 = xdomain[1] - box_pad
        y1 = ydomain[1] - box_pad
        for i, (label, color) in enumerate(items):
            y_top = y1 - i * row_gap
            fig.add_shape(
                type="rect",
                xref="paper", yref="paper",
                x0=x1 - swatch_w, x1=x1,
                y0=y_top - swatch_h, y1=y_top,
                line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
                fillcolor=color,
                layer="above",
            )
            fig.add_annotation(
                xref="paper", yref="paper",
                x=(x1 - swatch_w - 0.006),
                y=(y_top - swatch_h / 2.0),
                text=label,
                showarrow=False,
                xanchor="right",
                yanchor="middle",
                font=dict(size=font_size),
                bgcolor=("rgba(255,255,255,0.70)" if with_bg else None),
                bordercolor=("rgba(0,0,0,0.25)" if with_bg else None),
                borderwidth=(0.5 if with_bg else 0),
                align="left",
                opacity=0.98,
            )

    # --- Subplot titles ---
    title_row1 = "Time-Domain Signal (Original vs Reconstructed)"
    axis_hint = "Frequency axis" if "Frequency" in xaxis_label else "Period axis"
    if frac_info is not None:
        title_row2 = f"Power Spectrum (|FFT|², {axis_hint}) — Selected Band Energy = {frac_info:.1%}"
    else:
        title_row2 = f"Power Spectrum (|FFT|², {axis_hint})"

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{}], [{}]],
        vertical_spacing=0.12,
        subplot_titles=(title_row1, title_row2)
    )

    # ===== Row 1: Time-domain traces (disable global legend) =====
    fig.add_trace(go.Scatter(
        x=full_time, y=y_plot_original,
        name="Original",
        line=dict(color="gray"),
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=full_time, y=y_plot_recon,
        name="Reconstructed with selected frequencies",
        line=dict(color="blue"),
        showlegend=False
    ), row=1, col=1)

    # ===== Row 2: Power spectrum (two-layer bars) =====
    # Sort by x for proper bar ordering
    xa = np.asarray(x_vals_all, dtype=float)
    Pa = np.asarray(power_all, dtype=float)
    order_all = np.argsort(xa)
    xa = xa[order_all]
    Pa = Pa[order_all]

    # Light green: full spectrum
    if len(xa) > 1:
        mids = 0.5 * (xa[1:] + xa[:-1])
        edges = np.empty(len(xa) + 1, dtype=float)
        edges[1:-1] = mids
        edges[0] = xa[0] - (mids[0] - xa[0])
        edges[-1] = xa[-1] + (xa[-1] - mids[-1])
        eps = 1e-9
        edges = np.maximum.accumulate(edges)
        widths = np.clip(edges[1:] - edges[:-1], eps, None)
    else:
        widths = np.array([1.0])

    # Light green: full spectrum
    fig.add_trace(go.Bar(
        x=xa, y=Pa, width=widths,
        marker=dict(color='rgba(0,128,0,0.35)', line=dict(width=0)),
        name="Power (full, |FFT|²)",
        showlegend=False
    ), row=2, col=1)

    # Dark green overlay: selected band
    if band_mask is not None and np.any(band_mask):
        mask_sorted = np.asarray(band_mask)[order_all]
        fig.add_trace(go.Bar(
            x=xa[mask_sorted], y=Pa[mask_sorted], width=widths[mask_sorted],
            marker=dict(color='rgba(0,128,0,0.95)', line=dict(width=0)),
            name="Selected band",
            showlegend=False
        ), row=2, col=1)

    fig.update_layout(barmode="overlay", bargap=0.0, bargroupgap=0.0)

    # ===== Axes and layout =====
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text=xaxis_label, row=2, col=1)
    fig.update_yaxes(title_text="Power (|FFT|²)", type=yaxis_type, row=2, col=1)

    fig.update_layout(
        height=750,
        title=f"{signal}, {state.upper()}",
        title_font=dict(size=20),
        margin=dict(t=110, r=20, l=60, b=60),  # more top room so titles don’t get clipped
        showlegend=False
    )

    # ===== Make the official subplot titles larger =====
    # Plotly stores subplot_titles as annotations; bump sizes and tweak positions a bit.
    for ann in fig.layout.annotations:
        if ann.text == title_row1:
            ann.update(font=dict(size=16), xanchor="left", x=0.01)
        if ann.text == title_row2:
            ann.update(font=dict(size=16), xanchor="left", x=0.01)

    # Domains for legend placement
    x1d = tuple(fig.layout.xaxis.domain)
    y1d = tuple(fig.layout.yaxis.domain)
    x2d = tuple(fig.layout.xaxis2.domain)
    y2d = tuple(fig.layout.yaxis2.domain)

    add_internal_legend_topright(
        fig, x1d, y1d,
        items=[("Original", "gray"), ("Reconstructed", "blue")],
        font_size=12, with_bg=True
    )
    add_internal_legend_topright(
        fig, x2d, y2d,
        items=[("Power (full, |FFT|²)", "rgba(0,128,0,0.35)"),
               ("Selected band", "rgba(0,128,0,0.95)")],
        font_size=12, with_bg=True
    )

    return fig

def generate_fft_figure(
    df, state, signal, pad_length=0,
    low_cutoff=0.01, high_cutoff=0.5,
    filter_type="hard", pad_side="both",
    yaxis_type="linear",
    xaxis_mode="frequency",
):
    """
    Build the FFT figure with:
      - Row 1: time-domain original vs reconstructed
      - Row 2: power spectrum (full vs selected band)
    The spectrum x-axis can be 'frequency' (default) or 'period'; when 'period', we plot 1/f.
    """
    y_raw, t_raw, y_mean = extract_signal(df, state, signal)
    if y_raw is None:
        return go.Figure().update_layout(title="No data available")

    # Reduce points for speed/memory
    t_raw, y_raw = _truncate_and_decimate(t_raw, y_raw, max_points=5000)

    # Mean-removal for FFT
    y_for_fft = (y_raw - y_mean).astype("float32", copy=False)

    # Zero-padding for FFT
    y_padded_fft, full_time = preprocess_signal(y_for_fft, t_raw, int(pad_length or 0), pad_side)
    y_plot_original = pad_for_plot(y_for_fft, int(pad_length or 0), pad_side)

    # rFFT
    freqs_full, fft_vals_full = compute_fft(y_padded_fft)

    # Compute band energy fraction on positive frequencies (for subtitle)
    spectrum_full = np.abs(fft_vals_full) ** 2
    pos_mask_full = freqs_full > 0
    freqs_pos_all = freqs_full[pos_mask_full]
    spectrum_pos_all = spectrum_full[pos_mask_full]
    total_energy = float(np.sum(spectrum_pos_all))
    band_mask = (freqs_pos_all >= low_cutoff) & (freqs_pos_all <= high_cutoff)
    band_energy = float(np.sum(spectrum_pos_all[band_mask])) if total_energy > 0 else 0.0
    frac_energy = band_energy / total_energy if total_energy > 0 else 0.0

    # Hard-band filter in frequency domain
    filtered_fft_vals = apply_frequency_filter(freqs_full, fft_vals_full,
                                               float(low_cutoff), float(high_cutoff),
                                               method=filter_type)
    recon_signal = np.real(irfft(filtered_fft_vals, n=len(y_padded_fft))).astype("float32", copy=False)
    y_plot_recon = recon_signal

    # Positive-frequency power spectrum for plotting
    freqs_pos_display, power_display = compute_power_spectrum(freqs_full, filtered_fft_vals)
    # We also need the full-spectrum power (unfiltered) to show as the light-green base
    _, power_full_pos = compute_power_spectrum(freqs_full, fft_vals_full)

    # Select x-axis array and label for subplot 2
    if (xaxis_mode or "frequency") == "period":
        # Convert frequency to period (days). Avoid division by zero by using pos freqs only.
        x_vals_all = 1.0 / freqs_pos_all
        x_vals_display = 1.0 / freqs_pos_display
        x_label = "Period (days)"
    else:
        x_vals_all = freqs_pos_all
        x_vals_display = freqs_pos_display
        x_label = "Frequency (cycles/day)"

    # Prepare band mask aligned with x_vals_all
    band_mask_for_plot = band_mask

    # Convert timeline to strings for Plotly
    full_time_str = pd.to_datetime(full_time).strftime("%Y-%m-%d")

    # Render
    return create_fft_plot(
        full_time=full_time_str,
        y_plot_original=y_plot_original,
        y_plot_recon=y_plot_recon,
        x_vals_all=x_vals_all,              # base axis (full power background)
        power_all=power_full_pos,           # full power
        x_vals_display=x_vals_display,      # filtered spectrum axis (same grid usually)
        power_display=power_display,        # filtered power (not strictly needed if you only show full + band)
        band_mask=band_mask_for_plot,       # boolean mask on x_vals_all
        xaxis_label=x_label,
        signal=signal, state=state,
        yaxis_type=yaxis_type,
        frac_info=frac_energy,
    )

# ================= Factory App =================
def create_app(server, prefix="/app_fft_explorer/"):
    """
    Mounts the FFT explorer under the given prefix.
    NOTE: prefix MUST end with '/'.
    """
    if not prefix.endswith("/"):
        prefix = prefix + "/"

    PANEL_STYLE = {
        "marginBottom": "18px",
        "padding": "12px",
        "border": "1px solid #ddd",
        "borderRadius": "6px",
        "background": "#fafafa",
    }

    dash_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix=prefix,
        requests_pathname_prefix=prefix,
        suppress_callback_exceptions=True,
        title="FFT Time Series Explorer"
    )

    # -------- Layout --------
    dash_app.layout = html.Div(
        style={"display": "flex", "gap": "16px"},
        children=[
            # ================= Left column (28%): Controls + Simulated stacked =================
            html.Div(
                style={
                    "width": "28%",
                    "padding": "20px",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "16px",
                },
                children=[
                    # ----- Controls panel -----
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.H3("Controls", style={"marginTop": 0}),

                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    ["Drag and Drop or ", html.A("Select CSV File")]
                                ),
                                style={
                                    "width": "100%",
                                    "height": "60px",
                                    "lineHeight": "60px",
                                    "borderWidth": "1px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "marginBottom": "15px",
                                },
                                accept=".csv",
                                multiple=False,
                            ),
                            html.Div(
                                id="upload-status",
                                style={
                                    "fontSize": "12px",
                                    "marginBottom": "10px",
                                    "whiteSpace": "pre-wrap",
                                },
                            ),

                            # Hidden stores
                            dcc.Store(id="data-key"),
                            dcc.Store(id="schema-store"),

                            html.P(
                                "Workflow: (1) choose frequency band, (2) pad, (3) analyze.",
                                style={
                                    "fontSize": "13px",
                                    "marginBottom": "10px",
                                },
                            ),

                            html.Label("Select State:"),
                            dcc.Dropdown(
                                id="state-dropdown",
                                options=[],
                                value=None,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Select Signal:"),
                            dcc.Dropdown(
                                id="signal-dropdown",
                                options=[],
                                value=None,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Frequency Band (Low - High):"),
                            dcc.RangeSlider(
                                id="freq-range",
                                min=0.001,
                                max=0.5,
                                step=0.001,
                                value=[0.001, 0.5],
                                marks={0.01: "0.01", 0.1: "0.1", 0.3: "0.3", 0.5: "0.5"},
                                tooltip={"placement": "bottom"},
                            ),

                            html.Label("Pad Length (days):"),
                            dcc.Input(
                                id="pad-length",
                                type="number",
                                value=0,
                                min=0,
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Pad Side:"),
                            dcc.Dropdown(
                                id="pad-side",
                                options=[
                                    {"label": "Both", "value": "both"},
                                    {"label": "Left Only", "value": "left"},
                                ],
                                value="both",
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Y-axis scale (Amplitude):"),
                            dcc.Dropdown(
                                id="yaxis-scale",
                                options=[
                                    {"label": "Linear", "value": "linear"},
                                    {"label": "Log", "value": "log"},
                                ],
                                value="log",
                                style={"marginBottom": "14px"},
                            ),

                            html.Label("Spectrum X-axis:"),
                            dcc.Dropdown(
                                id="xaxis-mode",
                                options=[
                                    {"label": "Frequency (cycles/day)", "value": "frequency"},
                                    {"label": "Period (days)", "value": "period"},
                                ],
                                value="frequency",
                                style={"marginBottom": "14px"},
                            ),
                        ],
                    ),

                    # ----- Simulated data (optional) panel -----
                    html.Div(
                        style=PANEL_STYLE,
                        children=[
                            html.H4(
                                "Simulated data (optional)",
                                style={"marginTop": 0, "marginBottom": "8px"},
                            ),

                            html.Label("Type:"),
                            dcc.Dropdown(
                                id="synth-kind",
                                options=[
                                    {"label": "White Gaussian noise", "value": "white"},
                                    {"label": "AR(1) Gaussian noise", "value": "ar1"},
                                    {"label": "Poisson noise", "value": "poisson"},
                                    {"label": "Piecewise-constant + Gaussian noise", "value": "piecewise"},
                                    {"label": "Constant + spikes", "value": "spikes"},
                                ],
                                value="white",
                                style={"marginBottom": "10px"},
                            ),

                            html.Label("Length (days):"),
                            dcc.Input(
                                id="synth-length",
                                type="number",
                                value=730,
                                min=20,
                                step=1,
                                style={"marginBottom": "10px", "width": "100%"},
                            ),

                            html.Label("Random seed (optional):"),
                            dcc.Input(
                                id="synth-seed",
                                type="text",
                                value="",
                                style={"marginBottom": "10px", "width": "100%"},
                            ),

                            html.Div(
                                id="synth-param-block",
                                children=[
                                    # White Gaussian
                                    html.Div(
                                        id="params-white",
                                        children=[
                                            html.Label("Mean:"),
                                            dcc.Input(
                                                id="white-mean",
                                                type="number",
                                                value=0.0,
                                                step=0.1,
                                                style={"marginBottom": "6px", "width": "100%"},
                                            ),
                                            html.Label("Std:"),
                                            dcc.Input(
                                                id="white-std",
                                                type="number",
                                                value=1.0,
                                                step=0.1,
                                                style={"marginBottom": "10px", "width": "100%"},
                                            ),
                                        ],
                                    ),
                                    # AR(1)
                                    html.Div(
                                        id="params-ar1",
                                        style={"display": "none"},
                                        children=[
                                            html.Label("AR(1) phi:"),
                                            dcc.Input(
                                                id="ar1-phi",
                                                type="number",
                                                value=0.8,
                                                step=0.05,
                                                style={"marginBottom": "6px", "width": "100%"},
                                            ),
                                            html.Label("Innovation std:"),
                                            dcc.Input(
                                                id="ar1-std",
                                                type="number",
                                                value=1.0,
                                                step=0.1,
                                                style={"marginBottom": "6px", "width": "100%"},
                                            ),
                                            html.Label("Mean:"),
                                            dcc.Input(
                                                id="ar1-mean",
                                                type="number",
                                                value=0.0,
                                                step=0.1,
                                                style={"marginBottom": "10px", "width": "100%"},
                                            ),
                                        ],
                                    ),
                                    # Poisson
                                    html.Div(
                                        id="params-poisson",
                                        style={"display": "none"},
                                        children=[
                                            html.Label("Lambda (rate):"),
                                            dcc.Input(
                                                id="pois-lam",
                                                type="number",
                                                value=10.0,
                                                step=0.5,
                                                style={"marginBottom": "10px", "width": "100%"},
                                            ),
                                        ],
                                    ),
                                    # Piecewise
                                    html.Div(
                                        id="params-piecewise",
                                        style={"display": "none"},
                                        children=[
                                            html.Label("Segments:"),
                                            dcc.Input(
                                                id="pw-seg",
                                                type="number",
                                                value=4,
                                                step=1,
                                                style={"marginBottom": "6px", "width": "100%"},
                                            ),
                                            html.Label("Level range [low, high]:"),
                                            html.Div(
                                                style={"display": "flex", "gap": "8px", "marginBottom": "6px"},
                                                children=[
                                                    dcc.Input(
                                                        id="pw-low",
                                                        type="number",
                                                        value=0.0,
                                                        step=0.5,
                                                        style={"width": "50%"},
                                                    ),
                                                    dcc.Input(
                                                        id="pw-high",
                                                        type="number",
                                                        value=10.0,
                                                        step=0.5,
                                                        style={"width": "50%"},
                                                    ),
                                                ],
                                            ),
                                            html.Label("Noise std:"),
                                            dcc.Input(
                                                id="pw-noise",
                                                type="number",
                                                value=0.5,
                                                step=0.1,
                                                style={"marginBottom": "10px", "width": "100%"},
                                            ),
                                        ],
                                    ),
                                    # Spikes
                                    html.Div(
                                        id="params-spikes",
                                        style={"display": "none"},
                                        children=[
                                            html.Label("Baseline:"),
                                            dcc.Input(
                                                id="spk-base",
                                                type="number",
                                                value=0.0,
                                                step=0.5,
                                                style={"marginBottom": "6px", "width": "100%"},
                                            ),
                                            html.Label("Number of spikes:"),
                                            dcc.Input(
                                                id="spk-count",
                                                type="number",
                                                value=5,
                                                step=1,
                                                style={"marginBottom": "6px", "width": "100%"},
                                            ),
                                            html.Label("Spike amplitude range [low, high]:"),
                                            html.Div(
                                                style={"display": "flex", "gap": "8px", "marginBottom": "6px"},
                                                children=[
                                                    dcc.Input(
                                                        id="spk-alo",
                                                        type="number",
                                                        value=5.0,
                                                        step=0.5,
                                                        style={"width": "50%"},
                                                    ),
                                                    dcc.Input(
                                                        id="spk-ahi",
                                                        type="number",
                                                        value=20.0,
                                                        step=0.5,
                                                        style={"width": "50%"},
                                                    ),
                                                ],
                                            ),
                                            html.Label("Spike width [min, max] (days):"),
                                            html.Div(
                                                style={"display": "flex", "gap": "8px", "marginBottom": "10px"},
                                                children=[
                                                    dcc.Input(
                                                        id="spk-wmin",
                                                        type="number",
                                                        value=1,
                                                        step=1,
                                                        style={"width": "50%"},
                                                    ),
                                                    dcc.Input(
                                                        id="spk-wmax",
                                                        type="number",
                                                        value=5,
                                                        step=1,
                                                        style={"width": "50%"},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),

                            html.Button(
                                "Generate simulated dataset",
                                id="synth-generate",
                                n_clicks=0,
                                style={"width": "100%", "marginTop": "6px"},
                            ),
                            html.Div(
                                id="synth-status",
                                style={
                                    "fontSize": "12px",
                                    "marginTop": "6px",
                                    "whiteSpace": "pre-wrap",
                                    "color": "#444",
                                },
                            ),
                        ],
                    ),
                ],
            ),

            # ================= Right content (figure) =================
            html.Div(
                style={"width": "72%", "padding": "12px"},
                children=[dcc.Graph(id="fft-figure", style={"height": "720px"})],
            ),
        ],
    )

    # -------- Callbacks --------
    @dash_app.callback(
        Output("params-white", "style"),
        Output("params-ar1", "style"),
        Output("params-poisson", "style"),
        Output("params-piecewise", "style"),
        Output("params-spikes", "style"),
        Input("synth-kind", "value"),
    )
    def _toggle_synth_params(kind):
        # Show only the chosen block; others hidden
        show = {"display": "block"}
        hide = {"display": "none"}
        return (
            show if kind == "white" else hide,
            show if kind == "ar1" else hide,
            show if kind == "poisson" else hide,
            show if kind == "piecewise" else hide,
            show if kind == "spikes" else hide,
        )

    @dash_app.callback(
        Output("data-key", "data"),
        Output("schema-store", "data"),
        Output("upload-status", "children"),
        Output("synth-status", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        Input("synth-generate", "n_clicks"),
        State("synth-kind", "value"),
        State("synth-length", "value"),
        State("synth-seed", "value"),
        # White
        State("white-mean", "value"), State("white-std", "value"),
        # AR1
        State("ar1-phi", "value"), State("ar1-std", "value"), State("ar1-mean", "value"),
        # Poisson
        State("pois-lam", "value"),
        # Piecewise
        State("pw-seg", "value"), State("pw-low", "value"), State("pw-high", "value"), State("pw-noise", "value"),
        # Spikes
        State("spk-base", "value"), State("spk-count", "value"),
        State("spk-alo", "value"), State("spk-ahi", "value"),
        State("spk-wmin", "value"), State("spk-wmax", "value"),
        prevent_initial_call=False
    )
    def init_or_upload(contents, filename,
                       synth_clicks, kind, length, seed,
                       w_mean, w_std,
                       a_phi, a_std, a_mean,
                       p_lam,
                       pw_seg, pw_low, pw_high, pw_noise,
                       sp_base, sp_count, sp_alo, sp_ahi, sp_wmin, sp_wmax):
        """
        Unified initializer:
          - If 'Generate simuated dataset' was clicked, build simulated df and set it as current dataset.
          - Else if a CSV was uploaded, use the uploaded.
          - Else load the default CSV.
        """
        try:
            trig = ctx.triggered_id

            # ----- Case A: Simulated generator triggered -----
            if trig == "synth-generate" and (synth_clicks or 0) > 0:
                params = {}
                if kind == "white":
                    params = {"mean": float(w_mean or 0.0), "std": float(w_std or 1.0)}
                elif kind == "ar1":
                    params = {"phi": float(a_phi or 0.8), "std": float(a_std or 1.0), "mean": float(a_mean or 0.0)}
                elif kind == "poisson":
                    params = {"lam": float(p_lam or 10.0)}
                elif kind == "piecewise":
                    params = {
                        "n_segments": int(pw_seg or 4),
                        "low": float(pw_low or 0.0),
                        "high": float(pw_high or 10.0),
                        "noise_std": float(pw_noise or 0.5),
                    }
                elif kind == "spikes":
                    params = {
                        "baseline": float(sp_base or 0.0),
                        "n_spikes": int(sp_count or 5),
                        "amp_low": float(sp_alo or 5.0),
                        "amp_high": float(sp_ahi or 20.0),
                        "width_min": int(sp_wmin or 1),
                        "width_max": int(sp_wmax or 5),
                    }

                df_syn = make_simulated_df(kind, int(length or 365), seed, params)
                schema = detect_schema(df_syn)
                key = cache_put_df(schema["df"], tag=f"simulated:{kind}")
                status_upload = "Using simulated dataset.\n"
                status_synth = f"Generated simulated: kind={kind}, length={len(df_syn)}, seed={seed}\n"
                return (
                    key,
                    {"mode": schema["mode"], "states": schema["states"], "signals": schema["signals"]},
                    status_upload,
                    status_synth
                )

            # ----- Case B: CSV upload triggered -----
            if contents is not None and filename:
                df_up = parse_uploaded_contents(contents, filename)
                schema = detect_schema(df_up)
                key = cache_put_df(schema["df"], tag=filename)
                status_upload = f"Loaded file: {filename}\n"
                return (
                    key,
                    {"mode": schema["mode"], "states": schema["states"], "signals": schema["signals"]},
                    status_upload,
                    ""
                )

            # ----- Case C: Fallback to default dataset -----
            default_df = load_default_df()
            default_schema = detect_schema(default_df)
            key = cache_put_df(default_schema["df"], tag="default")
            status_upload = f"Using default dataset at {DEFAULT_PATH}\nMode: {default_schema['mode']}\nRows: {len(default_schema['df'])}"
            return (
                key,
                {"mode": default_schema["mode"], "states": default_schema["states"],
                 "signals": default_schema["signals"]},
                status_upload,
                ""
            )

        except Exception as e:
            # Fail-safe empty frame
            tiny = pd.DataFrame(
                {"time_value": pd.to_datetime([]), "geo_value": pd.Series([], dtype="category"), "value": []})
            key = cache_put_df(tiny, tag="empty")
            return key, {"mode": "simple", "states": ["uploaded"], "signals": ["value"]}, f"Error: {e}", ""

    @dash_app.callback(
        Output("state-dropdown", "options"),
        Output("state-dropdown", "value"),
        Output("signal-dropdown", "options"),
        Output("signal-dropdown", "value"),
        Input("schema-store", "data")
    )
    def populate_dropdowns(schema):
        if not schema:
            return [], None, [], None
        states = schema["states"]
        signals = schema["signals"]
        return ([{"label": str(s).upper(), "value": s} for s in states],
                (states[0] if states else None),
                [{"label": s, "value": s} for s in signals],
                (signals[0] if signals else None))

    @dash_app.callback(
        Output("fft-figure", "figure"),
        Input("data-key", "data"),
        Input("schema-store", "data"),
        Input("state-dropdown", "value"),
        Input("signal-dropdown", "value"),
        Input("freq-range", "value"),
        Input("pad-length", "value"),
        Input("pad-side", "value"),
        Input("yaxis-scale", "value"),
        Input("xaxis-mode", "value"),
    )
    def update_fft_plot(data_key, schema, state, signal, freq_range, pad_len, pad_side,
                        yaxis_scale, xaxis_mode):
        if not data_key or not schema or state is None or signal is None:
            return go.Figure().update_layout(title="No data available")

        df = cache_get_df(data_key)
        if df is None or df.empty:
            return go.Figure().update_layout(title="No data cached")

        if "time_value" not in df.columns or "geo_value" not in df.columns or signal not in df.columns:
            return go.Figure().update_layout(title="Uploaded data missing required columns after normalization.")

        low_cutoff, high_cutoff = (freq_range or [0.001, 0.5])

        return generate_fft_figure(
            df, state, signal,
            pad_length=int(pad_len) if pad_len is not None else 0,
            low_cutoff=float(low_cutoff), high_cutoff=float(high_cutoff),
            filter_type="hard", pad_side=pad_side,
            yaxis_type=yaxis_scale or "linear",
            xaxis_mode=xaxis_mode or "frequency",  # pass to generator
        )

    return dash_app