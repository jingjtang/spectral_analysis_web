# app_upload_fft.py  (revised with fixed graph height)
import base64
import io
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_PATH = "./data/ar_state.csv"

# ============== Data Loading ==============
def load_default_df():
    df = pd.read_csv(DEFAULT_PATH)
    df["time_value"] = pd.to_datetime(df["time_value"]).dt.strftime("%Y-%m-%d")
    return df

def parse_uploaded_contents(contents: str, filename: str) -> pd.DataFrame:
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except UnicodeDecodeError:
        df = pd.read_csv(io.StringIO(decoded.decode('latin-1')))
    for cand in ["time_value", "time", "date"]:
        if cand in df.columns:
            df[cand] = pd.to_datetime(df[cand], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def detect_schema(df: pd.DataFrame):
    if ("geo_value" in df.columns) and ("time_value" in df.columns):
        non_meta = [c for c in df.columns if c not in ["geo_value", "time_value"]]
        numeric_signals = [c for c in non_meta if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_signals) >= 1:
            out = df.copy()
            out["time_value"] = pd.to_datetime(out["time_value"], errors="coerce").dt.strftime("%Y-%m-%d")
            states = sorted(out["geo_value"].dropna().unique())
            signals = sorted(numeric_signals)
            return {"mode": "wide", "states": states, "signals": signals, "df": out}

    time_col = None
    for cand in ["time", "date", "time_value"]:
        if cand in df.columns:
            time_col = cand
            break
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    simple_df = pd.DataFrame()
    if time_col and len(numeric_cols) >= 1:
        signal_col = numeric_cols[0]
        simple_df["time_value"] = pd.to_datetime(df[time_col], errors="coerce").dt.strftime("%Y-%m-%d")
        simple_df["geo_value"] = "uploaded"
        simple_df["value"] = pd.to_numeric(df[signal_col], errors="coerce")
    elif not time_col and len(numeric_cols) >= 1:
        signal_col = numeric_cols[0]
        n = len(df)
        base = pd.Timestamp("2000-01-01")
        simple_df["time_value"] = [(base + pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n)]
        simple_df["geo_value"] = "uploaded"
        simple_df["value"] = pd.to_numeric(df[signal_col], errors="coerce")
    else:
        raise ValueError("Could not detect a usable schema.")
    return {"mode": "simple", "states": ["uploaded"], "signals": ["value"], "df": simple_df}

# ============== FFT Utils ==============
def extract_signal(df, state, signal, date_cutoff="2020-12-31"):
    curve = df.loc[
        (df["geo_value"] == state) & (df["time_value"] <= date_cutoff),
        ["time_value", signal]
    ].sort_values("time_value").dropna()
    if curve.empty:
        return None, None, None
    t_raw = pd.to_datetime(curve["time_value"])
    y_raw = curve[signal].astype(float).values
    y_raw[np.isnan(y_raw)] = 0.0
    y_mean = float(np.mean(y_raw)) if len(y_raw) else 0.0
    return y_raw, t_raw, y_mean

def preprocess_signal(y, t_raw, pad_length, pad_side="both"):
    if pad_side == "left":
        y_padded = np.concatenate([np.zeros(pad_length), y])
    else:
        y_padded = np.concatenate([np.zeros(pad_length), y, np.zeros(pad_length)])
    full_time = pd.date_range(
        start=t_raw.iloc[0] - pd.to_timedelta(pad_length, unit='D'),
        periods=len(y_padded), freq='D'
    )
    return y_padded, full_time

def pad_for_plot(y, pad_length, pad_side="both"):
    left = [0.0] * pad_length
    right = [0.0] * (pad_length if pad_side == "both" else 0)
    return np.array(left + list(y) + right, dtype=float)

def compute_fft(y_padded, dt=1.0):
    N = len(y_padded)
    freqs = fftfreq(N, d=dt)
    fft_vals = fft(y_padded)
    return freqs, fft_vals

def apply_frequency_filter(freqs, fft_vals, low_cutoff, high_cutoff, method="hard"):
    fft_filtered = fft_vals.copy()
    if method == "gaussian":
        eps = 1e-12 if high_cutoff == 0 else 0.0
        weights = np.exp(-((freqs) / (high_cutoff + eps)) ** 2)
        fft_filtered *= weights
    else:
        fft_filtered[(np.abs(freqs) < low_cutoff) | (np.abs(freqs) > high_cutoff)] = 0
    return fft_filtered

def compute_power_spectrum(freqs, fft_vals):
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask]) ** 2
    return pos_freqs, power

def compute_energy_distribution(freqs_full, fft_vals_full):
    """
    Returns: freqs_pos_sorted, pmf (energy PDF over positive freqs), cdf (cumulative energy share), df
    - Positive frequencies only (exclude f <= 0)
    - pmf[k] = P[k]/sum(P_pos), where P[k] = |Y[k]|^2
    - cdf = cumulative sum of pmf in ascending frequency
    """
    spectrum_full = np.abs(fft_vals_full) ** 2
    pos_mask = freqs_full > 0
    freqs_pos = freqs_full[pos_mask]
    P_pos = spectrum_full[pos_mask]

    if len(freqs_pos) == 0:
        return np.array([]), np.array([]), np.array([]), 1.0

    # sort by freq just in case
    order = np.argsort(freqs_pos)
    f_sorted = freqs_pos[order]
    P_sorted = P_pos[order]

    total = float(np.sum(P_sorted))
    if total <= 0:
        pmf = np.zeros_like(P_sorted)
        cdf = np.zeros_like(P_sorted)
    else:
        pmf = P_sorted / total               # "PDF" over discrete positive freq bins
        cdf = np.cumsum(pmf)                 # CDF from lowest positive freq upward

    # nominal bin width (uniform for FFT)
    df = float(np.mean(np.diff(f_sorted))) if len(f_sorted) > 1 else 1.0
    return f_sorted, pmf, cdf, df

def entropy_from_pdf(pdf_vals, mask=None):
    """
    Compute spectral entropy H = -sum p log p over positive-frequency PDF.
    Returns (H_total, H_band, H_norm, band_share).

    pdf_vals: array of p(f_k) over positive freqs (sum=1).
    mask: boolean array same length as pdf_vals selecting the user band.
    """
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
    """
    Coarsen the discrete PDF p(f) into nbins across frequency.
    Returns bin_centers, pdf_binned (sum of p(f) in each bin), bin_edges.
    If use_log is True, bins are uniform in log-frequency (exclude non-positive freqs beforehand).
    """
    if pdf_freqs is None or len(pdf_freqs) == 0:
        return np.array([]), np.array([]), np.array([])

    f = np.asarray(pdf_freqs)
    p = np.asarray(pdf_vals)
    mask = f > 0
    f = f[mask]
    p = p[mask]
    if len(f) == 0:
        return np.array([]), np.array([]), np.array([])

    if use_log:
        fmin, fmax = np.min(f), np.max(f)
        edges = np.logspace(np.log10(fmin), np.log10(fmax), nbins + 1)
    else:
        edges = np.linspace(np.min(f), np.max(f), nbins + 1)

    pdf_binned = np.zeros(nbins, dtype=float)
    # assign each freq to a bin
    idx = np.searchsorted(edges, f, side="right") - 1
    idx = np.clip(idx, 0, nbins - 1)
    for k in range(nbins):
        pdf_binned[k] = p[idx == k].sum()

    # bar x-locations = bin centers in linear freq
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    return bin_centers, pdf_binned, edges

def create_fft_plot(
    full_time, y_plot_original, y_plot_recon,
    freqs_power, power,                   # AFTER filtering, for Fig 2
    signal, state, yaxis_type="linear",
    frac_info=None,

    # Fig 3 inputs (from PRE-filter spectrum)
    pdf_freqs=None, pdf_vals=None, cdf_vals=None, df_bin=1.0,
    band_mask_pdf=None,
    pdf_binned_freqs=None, pdf_binned_vals=None, pdf_yaxis_type="linear",

    # NEW: entropy info to show in Fig 3 subtitle
    H_total=None, H_band=None, H_norm=None, band_share=None
):
    subtitle_power = "FFT Amplitude Spectrum (Frequency Domain)"
    if frac_info is not None:
        subtitle_power += f"<br> — Selected Band Energy = {frac_info:.1%}"

    # Build entropy subtitle text (plain text so it renders)
    ent_bits = []
    if H_total is not None:
        ent_bits.append(f"H={H_total:.3f}")
    if H_norm is not None:
        ent_bits.append(f"H_norm={H_norm:.3f}")
    if band_share is not None:
        ent_bits.append(f"Entropy Fraction={band_share:.1%}")
    ent_txt = (" - " + " ".join(ent_bits)) if ent_bits else ""

    subtitle_pdf = (
        "Energy PDF (bars, binned) & CDF (line)<br>"
        + ent_txt
    )

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2}, None],
               [{}, {"secondary_y": True}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        subplot_titles=[
            "Time-Domain Signal (Original vs Reconstructed)",
            "Power Spectrum (After Filtering)",
            "Energy PDF & CDF (Positive Frequencies)"
        ]
    )

    # --- Row 1: Time series ---
    fig.add_trace(go.Scatter(x=full_time, y=y_plot_original,
                             name="Original (mean-removed)", line=dict(color="gray")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=full_time, y=y_plot_recon,
                             name="Reconstructed (filtered, mean-removed)", line=dict(color="blue")),
                  row=1, col=1)

    # --- Row 2, Col 1: Power (after filtering) ---
    order = np.argsort(freqs_power)
    f2 = freqs_power[order]; P2 = power[order]
    if len(f2) > 1:
        df_local = float(np.mean(np.diff(f2))); widths = np.full_like(f2, df_local)
    else:
        widths = np.array([0.0])
    fig.add_trace(go.Bar(x=f2, y=P2, width=widths,
                         marker=dict(color='rgba(0,128,0,0.4)'), name="Power"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=f2, y=P2, mode='markers',
                             marker=dict(size=4, color='green'), showlegend=False),
                  row=2, col=1)

    # --- Row 2, Col 2: PDF (binned) + CDF ---
    if pdf_binned_freqs is not None and len(pdf_binned_freqs) > 0 and pdf_binned_vals is not None:
        x_pdf = pdf_binned_freqs; y_pdf = pdf_binned_vals
        if len(x_pdf) > 1:
            edges = np.zeros(len(x_pdf) + 1, dtype=float)
            edges[1:-1] = 0.5 * (x_pdf[:-1] + x_pdf[1:])
            edges[0]  = x_pdf[0]  - (edges[1] - x_pdf[0])
            edges[-1] = x_pdf[-1] + (x_pdf[-1] - edges[-2])
            widths_pdf = edges[1:] - edges[:-1]
        else:
            widths_pdf = np.array([df_bin])
        fig.add_trace(go.Bar(x=x_pdf, y=y_pdf, width=widths_pdf,
                             name="Energy PDF p(f) (binned)",
                             marker=dict(color='rgba(0,0,255,0.45)')),
                      row=2, col=2, secondary_y=False)
    elif pdf_freqs is not None and len(pdf_freqs) > 0 and pdf_vals is not None:
        widths_pdf = np.full_like(pdf_freqs, df_bin) if len(pdf_freqs) > 1 else np.array([0.0])
        fig.add_trace(go.Bar(x=pdf_freqs, y=pdf_vals, width=widths_pdf,
                             name="Energy PDF p(f)", marker=dict(color='rgba(0,0,255,0.35)')),
                      row=2, col=2, secondary_y=False)

    # if band_mask_pdf is not None and pdf_freqs is not None and len(pdf_freqs) > 0:
    #     fig.add_trace(go.Bar(x=pdf_freqs[band_mask_pdf], y=pdf_vals[band_mask_pdf],
    #                          name="Selected band (PDF)",
    #                          marker=dict(line=dict(width=1.0), opacity=0.9)),
    #                   row=2, col=2, secondary_y=False)

    if band_mask_pdf is not None and len(pdf_freqs) > 0:
        fig.add_vline(
            x=pdf_freqs[band_mask_pdf].min(),
            line=dict(color="red", dash="dash"),
            row=2, col=2
        )
        fig.add_vline(
            x=pdf_freqs[band_mask_pdf].max(),
            line=dict(color="red", dash="dash"),
            row=2, col=2
        )

    if cdf_vals is not None and pdf_freqs is not None and len(pdf_freqs) == len(cdf_vals):
        fig.add_trace(go.Scatter(x=pdf_freqs, y=cdf_vals, mode='lines+markers',
                                 name="CDF (cumulative energy share)"),
                      row=2, col=2, secondary_y=True)

    # Axes & layout
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (cycles/day)", row=2, col=1)
    fig.update_yaxes(title_text="Power", type=yaxis_type, row=2, col=1)

    fig.update_xaxes(title_text="Frequency (cycles/day)", row=2, col=2)
    fig.update_yaxes(title_text="PDF p(f)", type=pdf_yaxis_type, row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="CDF", range=[0, 1], row=2, col=2, secondary_y=True)

    fig.update_layout(
        height=900,
        title=f"{signal} in {state.upper()} — FFT Analysis",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        annotations=[
            dict(text="Time-Domain Signal (Original vs Reconstructed)",
                 xref="paper", yref="paper", x=0.0, y=1.085, showarrow=False, font=dict(size=13)),
            dict(text=subtitle_power,
                 xref="paper", yref="paper", x=0.12, y=0.46, showarrow=False, font=dict(size=12)),
            dict(text=subtitle_pdf,   # <— shows entropy here
                 xref="paper", yref="paper", x=0.88, y=0.46, showarrow=False,
                 font=dict(size=12), xanchor="right")
        ]
    )
    return fig


def generate_fft_figure(
    df, state, signal, pad_length=0,
    low_cutoff=0.01, high_cutoff=0.5,
    filter_type="hard", pad_side="both",
    yaxis_type="linear",
    pdf_nbins=40, pdf_logbins=False, pdf_yaxis_type="linear"
):
    y_raw, t_raw, y_mean = extract_signal(df, state, signal)
    if y_raw is None:
        return go.Figure().update_layout(title="No data available")

    y_for_fft = y_raw - y_mean
    y_padded_fft, full_time = preprocess_signal(y_for_fft, t_raw, pad_length, pad_side)
    y_plot_original = pad_for_plot(y_for_fft, pad_length, pad_side)

    freqs_full, fft_vals_full = compute_fft(y_padded_fft)

    spectrum_full = np.abs(fft_vals_full) ** 2
    pos_mask_full = freqs_full > 0
    freqs_pos_all = freqs_full[pos_mask_full]
    spectrum_pos_all = spectrum_full[pos_mask_full]
    total_energy = float(np.sum(spectrum_pos_all))
    band_mask = (freqs_pos_all >= low_cutoff) & (freqs_pos_all <= high_cutoff)
    band_energy = float(np.sum(spectrum_pos_all[band_mask])) if total_energy > 0 else 0.0
    frac_energy = band_energy / total_energy if total_energy > 0 else 0.0

    # PDF/CDF over positive freqs (pre-filter)
    pdf_freqs, pdf_vals, cdf_vals, df_bin = compute_energy_distribution(freqs_full, fft_vals_full)
    band_mask_pdf = (pdf_freqs >= low_cutoff) & (pdf_freqs <= high_cutoff) if len(pdf_freqs) else None

    # --- NEW: entropy on the PDF
    H_total, H_band, H_norm, band_share = entropy_from_pdf(pdf_vals, mask=band_mask_pdf)

    # BIN the PDF for visibility
    pdf_binned_freqs, pdf_binned_vals, _ = bin_energy_pdf(
        pdf_freqs, pdf_vals, nbins=int(pdf_nbins), use_log=bool(pdf_logbins)
    )

    # Filter for reconstruction
    filtered_fft_vals = apply_frequency_filter(freqs_full, fft_vals_full,
                                               low_cutoff, high_cutoff,
                                               method=filter_type)
    recon_signal = np.real(ifft(filtered_fft_vals))
    y_plot_recon = recon_signal

    freqs_pos_display, power_display = compute_power_spectrum(freqs_full, filtered_fft_vals)

    return create_fft_plot(
        full_time=full_time,
        y_plot_original=y_plot_original,
        y_plot_recon=y_plot_recon,
        freqs_power=freqs_pos_display,
        power=power_display,
        signal=signal, state=state,
        yaxis_type=yaxis_type,
        frac_info=frac_energy,
        pdf_freqs=pdf_freqs, pdf_vals=pdf_vals, cdf_vals=cdf_vals, df_bin=df_bin,
        band_mask_pdf=band_mask_pdf,
        pdf_binned_freqs=pdf_binned_freqs, pdf_binned_vals=pdf_binned_vals,
        pdf_yaxis_type=pdf_yaxis_type,
        # NEW: pass entropy to subtitle of Fig 3
        H_total=H_total, H_band=H_band, H_norm=H_norm, band_share=band_share
    )

# ================= Factory App =================
def create_app(server, prefix="/app_fft_upload/"):
    """
    Mounts the FFT upload explorer under the given prefix.
    NOTE: prefix MUST end with '/'.
    """
    if not prefix.endswith("/"):
        prefix = prefix + "/"

    dash_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix=prefix,
        requests_pathname_prefix=prefix,
        suppress_callback_exceptions=True,
        title="FFT Time Series Explorer"
    )

    # -------- Layout --------
    dash_app.layout = html.Div(style={"display": "flex"}, children=[
        html.Div(style={"width": "28%", "padding": "20px"}, children=[
            html.H3("Controls"),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select CSV File')]),
                style={'width': '100%', 'height': '60px', 'lineHeight': '60px',
                       'borderWidth': '1px', 'borderStyle': 'dashed',
                       'borderRadius': '5px', 'textAlign': 'center', 'marginBottom': '15px'},
                accept='.csv', multiple=False
            ),
            html.Div(id='upload-status', style={"fontSize": "12px", "marginBottom": "10px", "whiteSpace": "pre-wrap"}),
            dcc.Store(id="data-store"),
            dcc.Store(id="schema-store"),
            html.P("Workflow: (1) choose frequency band, (2) pad, (3) analyze.",
                   style={"fontSize": "13px", "marginBottom": "10px"}),
            html.Label("Select State:"), dcc.Dropdown(id="state-dropdown", options=[], value=None, style={"marginBottom": "14px"}),
            html.Label("Select Signal:"), dcc.Dropdown(id="signal-dropdown", options=[], value=None, style={"marginBottom": "14px"}),
            html.Label("Frequency Band (Low - High):"),
            dcc.RangeSlider(id="freq-range", min=0.001, max=0.5, step=0.001, value=[0.001, 0.5],
                            marks={0.01: "0.01", 0.1: "0.1", 0.3: "0.3", 0.5: "0.5"},
                            tooltip={"placement": "bottom"}),
            html.Label("Pad Length (days):"),
            dcc.Input(id="pad-length", type="number", value=0, min=0, style={"marginBottom": "14px"}),
            html.Label("Pad Side:"),
            dcc.Dropdown(id="pad-side",
                         options=[{"label": "Both", "value": "both"}, {"label": "Left Only", "value": "left"}],
                         value="both", style={"marginBottom": "14px"}),
            html.Label("Y-axis scale (Amplitude):"),
            dcc.Dropdown(
                id="yaxis-scale",
                options=[{"label": "Linear", "value": "linear"},
                         {"label": "Log", "value": "log"}],
                value="log",
                style={"marginBottom": "14px"}
            ),
            html.Label("PDF bins (Figure 3):"),
            dcc.Slider(id="pdf-nbins", min=10, max=120, step=5, value=40,
                       marks={10: "10", 40: "40", 80: "80", 120: "120"},
                       tooltip={"placement": "bottom"}),
            html.Label("PDF frequency binning:"),
            dcc.Dropdown(id="pdf-logbins",
                         options=[{"label": "Linear frequency bins", "value": "linear"},
                                  {"label": "Log-frequency bins", "value": "log"}],
                         value="linear", style={"marginBottom": "14px"}),
            html.Label("PDF y-axis (Figure 3):"),
            dcc.Dropdown(id="pdf-yaxis-scale",
                         options=[{"label": "Linear", "value": "linear"},
                                  {"label": "Log", "value": "log"}],
                         value="log", style={"marginBottom": "14px"}),
        ]),

        html.Div(style={"width": "72%", "padding": "20px"}, children=[
            html.P("The top plot always shows the mean-removed original (gray) and the mean-removed reconstruction (blue). "
                   "Padding is shown explicitly as zeros on the original curve.",
                   style={"fontSize": "14px", "marginBottom": "10px"}),
            dcc.Graph(id="fft-figure", style={"height": "750px"})
        ])
    ])

    # -------- Callbacks --------
    @dash_app.callback(
        Output("data-store", "data"),
        Output("schema-store", "data"),
        Output("upload-status", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        prevent_initial_call=False
    )
    def init_or_upload(contents, filename):
        if contents is not None and filename:
            try:
                df_up = parse_uploaded_contents(contents, filename)
                schema = detect_schema(df_up)
                status = f"Loaded file: {filename}\n"
                return (schema["df"].to_dict("records"),
                        {"mode": schema["mode"], "states": schema["states"], "signals": schema["signals"]},
                        status)
            except Exception as e:
                default_df = load_default_df()
                default_schema = detect_schema(default_df)
                status = f"Failed to parse '{filename}': {e}\nLoaded default data instead."
                return (default_schema["df"].to_dict("records"),
                        {"mode": default_schema["mode"], "states": default_schema["states"], "signals": default_schema["signals"]},
                        status)
        default_df = load_default_df()
        default_schema = detect_schema(default_df)
        status = f"Using default dataset at {DEFAULT_PATH}\nMode: {default_schema['mode']}\nRows: {len(default_schema['df'])}"
        return (default_schema["df"].to_dict("records"),
                {"mode": default_schema["mode"], "states": default_schema["states"], "signals": default_schema["signals"]},
                status)

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
        return ([{"label": s.upper(), "value": s} for s in states],
                (states[0] if states else None),
                [{"label": s, "value": s} for s in signals],
                (signals[0] if signals else None))

    @dash_app.callback(
        Output("fft-figure", "figure"),
        Input("data-store", "data"),
        Input("schema-store", "data"),
        Input("state-dropdown", "value"),
        Input("signal-dropdown", "value"),
        Input("freq-range", "value"),
        Input("pad-length", "value"),
        Input("pad-side", "value"),
        Input("yaxis-scale", "value"),
        Input("pdf-nbins", "value"),
        Input("pdf-logbins", "value"),
        Input("pdf-yaxis-scale", "value"),
    )
    def update_fft_plot(data_records, schema, state, signal, freq_range, pad_len, pad_side,
                        yaxis_scale, pdf_nbins, pdf_logbins, pdf_yaxis_scale):
        if not data_records or not schema or state is None or signal is None:
            return go.Figure().update_layout(title="No data available")
        df = pd.DataFrame(data_records).copy()
        if "time_value" not in df.columns or "geo_value" not in df.columns or signal not in df.columns:
            return go.Figure().update_layout(title="Uploaded data missing required columns after normalization.")
        low_cutoff, high_cutoff = freq_range
        return generate_fft_figure(
            df, state, signal,
            pad_length=int(pad_len) if pad_len is not None else 0,
            low_cutoff=float(low_cutoff), high_cutoff=float(high_cutoff),
            filter_type="hard", pad_side=pad_side,
            yaxis_type=yaxis_scale,
            pdf_nbins=int(pdf_nbins) if pdf_nbins is not None else 40,
            pdf_logbins=(pdf_logbins == "log"),
            pdf_yaxis_type=pdf_yaxis_scale or "linear"
        )

    return dash_app