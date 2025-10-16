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
            Our research explores how viral transmission changes over time and what drives those changes.
            Candidate factors include weather, mobility, demographics, and viral evolution. Some, such as
            daily temperature or movement patterns, vary rapidly—but whether such fine-scale fluctuations
            truly shape epidemic dynamics remains uncertain.
          </p>
          <p>
            The time-varying reproductive number, <i>R₀(t)</i>, often looks very different across estimation methods:
            some curves show smooth seasonal waves, while others oscillate day to day. This raises a key
            question—are these rapid swings real, or artifacts of how surveillance data are filtered through
            reporting delays?
          </p>
          <p>
            Observed data such as cases, hospitalizations, and deaths are delayed reflections of infections,
            each shaped by a reporting-delay distribution that acts as a low-pass filter—preserving long-term
            trends but suppressing short-term variability. As a result, distinct infection curves can produce
            nearly identical observed signals, creating a fundamental identifiability problem.
          </p>
          <p>
            These interactive demos quantify how much high-frequency information survives—or is lost—when epidemic
            signals are filtered by reporting delays. Together, they help reveal which aspects of transmission
            dynamics can be reliably inferred from data.
          </p>

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


