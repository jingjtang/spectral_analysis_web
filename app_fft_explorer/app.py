# app_upload_fft.py  (revised with fixed graph height)
import base64
import io
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft, fftfreq
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache
import uuid
from flask import request
from flask_caching import Cache

# simple server-side cache (in-memory)
cache = Cache(config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

DEFAULT_PATH = "./data/ar_state.csv"

# ============== Data Loading ==============
@lru_cache(maxsize=8)
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
    magnitude = np.abs(fft_vals[pos_mask])
    periods = 1.0 / pos_freqs
    return periods, magnitude

def create_fft_plot(full_time, y_plot_original, y_plot_recon,
                    periods, magnitude, signal, state):
    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.15,
        subplot_titles=[
            "Time-Domain Signal (Original vs Reconstructed)",
            "FFT Amplitude Spectrum (Period Domain)"
        ]
    )
    fig.add_trace(go.Scatter(x=full_time, y=y_plot_original,
                             name="Original (mean-removed)", line=dict(color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=full_time, y=y_plot_recon,
                             name="Reconstructed (filtered, mean-removed)", line=dict(color="blue")), row=1, col=1)

    sorted_idx = np.argsort(periods)
    sorted_periods = periods[sorted_idx]
    sorted_magnitude = magnitude[sorted_idx]
    bar_widths = np.gradient(sorted_periods)

    fig.add_trace(go.Bar(
        x=sorted_periods, y=sorted_magnitude, width=bar_widths,
        marker=dict(color='rgba(0,128,0,0.4)'), name="Amplitude"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(x=sorted_periods, y=sorted_magnitude,
                             mode='markers', marker=dict(size=4, color='green'),
                             showlegend=False), row=2, col=1)

    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_xaxes(title_text="Period (days)", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", type="log", row=2, col=1)

    # 👇 固定高度，避免页面无限拉长
    fig.update_layout(
        height=700,
        title=f"{signal} in {state.upper()} — FFT Analysis",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def generate_fft_figure(df, state, signal, pad_length=10,
                        low_cutoff=0.01, high_cutoff=0.5,
                        filter_type="hard", pad_side="both"):
    y_raw, t_raw, y_mean = extract_signal(df, state, signal)
    if y_raw is None:
        return go.Figure().update_layout(title="No data available")
    y_for_fft = y_raw - y_mean
    y_padded_fft, full_time = preprocess_signal(y_for_fft, t_raw, pad_length, pad_side)
    y_plot_original = pad_for_plot(y_for_fft, pad_length, pad_side)
    freqs, fft_vals = compute_fft(y_padded_fft)
    filtered_fft_vals = apply_frequency_filter(freqs, fft_vals,
                                               low_cutoff, high_cutoff,
                                               method=filter_type)
    recon_signal = np.real(ifft(filtered_fft_vals))
    y_plot_recon = recon_signal
    periods, magnitude = compute_power_spectrum(freqs, filtered_fft_vals)
    return create_fft_plot(full_time, y_plot_original, y_plot_recon,
                           periods, magnitude, signal, state)

# ================= Factory App =================
def create_app(server, prefix="/app_fft_upload/"):
    cache.init_app(server)  # <-- init cache

    dash_app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix=prefix,
        requests_pathname_prefix=prefix,
        suppress_callback_exceptions=True,
        title="FFT Time Series Explorer"
    )

    dash_app.layout = html.Div(style={"display": "flex"}, children=[

        html.Div(style={"width": "28%", "padding": "20px"}, children=[
            html.H3("Controls"),
            dcc.Store(id="session-id"),
            dcc.Store(id="data-key"),
            dcc.Store(id="schema-store"),  # keep only small schema in browser
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
            dcc.Input(id="pad-length", type="number", value=10, min=0, style={"marginBottom": "14px"}),
            html.Label("Pad Side:"),
            dcc.Dropdown(id="pad-side",
                         options=[{"label": "Both", "value": "both"}, {"label": "Left Only", "value": "left"}],
                         value="both", style={"marginBottom": "14px"}),
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
        Output("session-id", "data"),
        Input("session-id", "data"),
        prevent_initial_call=False
    )
    def ensure_session_id(existing):
        if existing:
            return existing
        # tie to client if you want: request.remote_addr etc.
        return str(uuid.uuid4())

    @dash_app.callback(
        Output("data-key", "data"),
        Output("schema-store", "data"),
        Output("upload-status", "children"),
        Input("upload-data", "contents"),
        State("upload-data", "filename"),
        State("session-id", "data"),
        prevent_initial_call=False
    )
    def init_or_upload(contents, filename, sid):
        def downcast(df):
            # shrink memory: dates → str, categorical for geo_value, float32 for values
            if "geo_value" in df.columns:
                df["geo_value"] = df["geo_value"].astype("category")
            for c in df.columns:
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
                elif pd.api.types.is_integer_dtype(df[c]):
                    df[c] = pd.to_numeric(df[c], errors="coerce", downcast="integer")
            return df

        try:
            if contents and filename:
                df_up = parse_uploaded_contents(contents, filename)
                df_up = downcast(df_up)
                schema = detect_schema(df_up)
                key = f"{sid}:uploaded"
                cache.set(key, schema["df"], timeout=3600)  # store df only on server
                # send only schema (small) to client
                return key, {"mode": schema["mode"], "states": schema["states"],
                             "signals": schema["signals"]}, f"Loaded file: {filename}\n"
            else:
                df = load_default_df()
                df = downcast(df)
                schema = detect_schema(df)
                key = f"{sid}:default"
                cache.set(key, schema["df"], timeout=3600)
                return key, {"mode": schema["mode"], "states": schema["states"], "signals": schema["signals"]}, \
                    f"Using default dataset at {DEFAULT_PATH}\nMode: {schema['mode']}\nRows: {len(schema['df'])}"
        except Exception as e:
            # robust fallback to default
            df = load_default_df()
            df = downcast(df)
            schema = detect_schema(df)
            key = f"{sid}:default"
            cache.set(key, schema["df"], timeout=3600)
            return key, {"mode": schema["mode"], "states": schema["states"], "signals": schema["signals"]}, \
                f"Failed to parse '{filename}': {e}\nLoaded default data instead."

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
        Input("data-key", "data"),  # <- use key, not records
        Input("schema-store", "data"),
        Input("state-dropdown", "value"),
        Input("signal-dropdown", "value"),
        Input("freq-range", "value"),
        Input("pad-length", "value"),
        Input("pad-side", "value"),
    )
    def update_fft_plot(data_key, schema, state, signal, freq_range, pad_len, pad_side):
        if not data_key or not schema or state is None or signal is None:
            return go.Figure().update_layout(title="No data available")

        df = cache.get(data_key)
        if df is None:
            return go.Figure().update_layout(title="Data expired from cache. Please reload.")

        # IMPORTANT: avoid df.copy(); work on df directly or on small slices
        if "time_value" not in df.columns or "geo_value" not in df.columns or signal not in df.columns:
            return go.Figure().update_layout(title="Uploaded data missing required columns after normalization.")

        low_cutoff, high_cutoff = freq_range
        return generate_fft_figure(
            df, state, signal,
            pad_length=int(pad_len) if pad_len is not None else 0,
            low_cutoff=float(low_cutoff), high_cutoff=float(high_cutoff),
            filter_type="hard", pad_side=pad_side
        )

    return dash_app


# ============ Optional local debug ============
# if __name__ == "__main__":
#     from flask import Flask
#     server = Flask(__name__)
#     app = create_app(server, prefix="/app_fft_upload/")
#     app.run_server(host="0.0.0.0", port=8054, debug=True)