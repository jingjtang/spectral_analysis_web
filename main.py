# main.py
import os
from flask import Flask, render_template_string

from app_delay_filtering.app import create_app as create_delay_app
from app_fft_explorer.app import create_app as create_fft_app
# from app_multiple_delay_filtering.app import create_app as create_multi_app

def create_server():
    server = Flask(__name__)

    #  main page
    @server.route("/")
    def index():
        return render_template_string("""
        <html>
        <head>
            <title>COVID Signal Lab — Demos</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
              body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:880px;margin:40px auto;padding:0 16px}
              .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-top:16px}
              a.card{display:block;border:1px solid #e5e7eb;border-radius:12px;padding:16px;text-decoration:none;color:#111}
              a.card:hover{border-color:#111}
              h1{font-size:28px;margin:0 0 8px}
              p{margin:6px 0 0;color:#444}
            </style>
        </head>
        <body>
          <h1>COVID Delay & FFT Apps</h1>
          <p>First App：</p>
          <div class="grid">
            <a class="card" href="/app_delay_filtering/">
              <strong>Delay Filtering</strong>
              <p>delay_filtering</p>
            </a>
            <a class="card" href="/app_fft_explorer/">
              <strong>FFT Explorer</strong>
              <p>fft_transformation</p>
            </a>
            <a class="card" href="/app_multiple_delay_filtering/">
              <strong>Multiple Delay Filtering</strong>
              <p>multiple_delay_filtering</p>
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


