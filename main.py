# main.py
import os
from flask import Flask, render_template_string

from app_delay_filtering.app import create_app as create_delay_app
from app_fft_explorer.app import create_app as create_fft_app
# from app_multiple_delay_filtering.app import create_app as create_multi_app
from flask import Flask
from flask_caching import Cache

cache = Cache(config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 3600})

def create_server():
    server = Flask(__name__)
    cache.init_app(server)
    #  main page
    @server.route("/")
    def index():
        return render_template_string("""
        <html>
        <head>
            <title>COVID Signal Lab — Demos</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body {
                font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
                max-width: 880px;
                margin: 40px auto;
                padding: 0 16px;
              }
              .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                gap: 16px;
                margin-top: 24px;
              }
              a.card {
                display: block;
                border: 1px solid #e5e7eb;
                border-radius: 12px;
                padding: 16px;
                text-decoration: none;
                color: #111;
                background: #fafafa;
                transition: all 0.2s ease;
              }
              a.card:hover {
                border-color: #111;
                background: #f3f4f6;
              }
              h1 {
                font-size: 28px;
                margin: 0 0 12px;
              }
              p {
                margin: 8px 0;
                color: #444;
                line-height: 1.5;
              }
              hr {
                border: none;
                border-top: 1px solid #e5e7eb;
                margin: 24px 0;
              }
            </style>
        </head>
        <body>
            <h1>Spectral Analysis Apps</h1>
            
            <p>
              Our goal is to understand the drivers of viral transmission and how they evolve over time. Potential drivers include weather variability, human mobility, demographic context, and viral evolution. Some, such as daily temperature, humidity, and mobility flows, are measured at high temporal resolution and capture day-to-day changes. Yet whether such fine-scale fluctuations genuinely translate into observable epidemic impacts—and through what mechanisms—remains uncertain.
            </p>
            
            <p>
              The time-varying reproductive number, <em>R<sub>0</sub>(t)</em>, is the standard measure of transmissibility, but its estimated trajectories differ widely across methods. Some appear as smooth seasonal waves (e.g., R0-CovidEstim), while others fluctuate sharply on a daily scale (e.g., R0-CU). This discrepancy raises a central question: is the true <em>R<sub>0</sub>(t)</em> inherently volatile, or does the structure of surveillance data make such volatility fundamentally unobservable?
            </p>
            
            <p>
              Infections themselves are not directly observed. Instead, reported cases, hospitalizations, and deaths are delayed and distorted reflections of infection incidence, each generated through convolution with a reporting-delay distribution. In the frequency domain, such convolutions act as low-pass filters that preserve slow, long-term variations but suppress rapid oscillations. Consequently, even highly dynamic infection processes may appear smoothed in the observed data, creating an identifiability problem in which distinct upstream infection curves yield nearly indistinguishable downstream signals.
            </p>
            
            <p>
              The main objective of this project is to quantitatively characterize the information loss introduced by reporting delays. Specifically, we aim to:
            </p>
            
            <ul>
              <li>
                <strong>Provide a principled analytical framework:</strong>
                Move beyond qualitative intuition toward a quantitative, frequency-based understanding of the identifiability constraints that shape epidemic inference and model interpretation.
              </li>
              <li>
                <strong>Decompose signals by temporal scale:</strong>
                Use frequency-domain analysis to separate each signal into its component frequencies, allowing us to assess which temporal scales of variation remain statistically recoverable from observed data.
              </li>
              <li>
                <strong>Measure high-frequency information retention & Quantify information loss:</strong>
                Evaluate how much of the high-frequency content in epidemic signals survives after being filtered through the reporting-delay distribution, and determine what proportion of variation—particularly in the high-frequency range—is irretrievably lost during convolution, leading to nearly indistinguishable downstream observations.
              </li>
              <li>
                <strong>Incorporate the role of noise:</strong>
                Noise fundamentally shapes the severity of identifiability constraints. Delay distributions invariably suppress high-frequency components of upstream processes, but the extent to which this matters in practice depends on the signal-to-noise regime of the observed data. At high noise levels, short-period fluctuations in surveillance series are dominated by stochastic variability, making delay-induced attenuation less consequential. By contrast, when noise is moderate, the smoothing imposed by delays becomes the primary source of information loss.
              </li>
            </ul>
            
            <hr>

          <div class="grid">
            <a class="card" href="/app_fft_explorer/">
              <strong>FFT Explorer</strong>
              <p>Decompose epidemic time series into frequency components to study information loss.</p>
            </a>
            <a class="card" href="/app_delay_filtering/">
              <strong>Delay Filtering</strong>
              <p>Visualize how delay distributions act as low-pass filters on epidemic signals.</p>
            </a>
            <a class="card" href="/app_multiple_delay_filtering/">
              <strong>Multiple Delay Filtering</strong>
              <p>Compare the smoothing effects of different reporting-delay distributions.</p>
            </a>
          </div>
        </body>
        </html>
        """)

    create_delay_app(server, prefix="/app_delay_filtering/")
    create_fft_app(server,   prefix="/app_fft_explorer/")
    # create_multi_app(server, prefix="/app_multiple_delay_filtering/")

    return server

server = create_server()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    server.run(host="0.0.0.0", port=port, debug=True)


