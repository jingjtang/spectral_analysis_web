# Multi-App Dashboard Suite for Spectral Analysis

This repository contains a collection of independent Dash applications for interactive time-series and spectral analysis.
Each app is fully self-contained: you can run them separately without needing to launch the entire suite.

## Available Apps
```
App A: FFT Transformation — app_fft_transformation/
```
Explore frequency-domain properties of time series with interactive FFT filtering.

```
App B: Delay Filtering — app_delay_filtering/
```
Simulate symptomatic-to-reporting processes using configurable delay distributions and explore how delays act as low-pass filters on epidemic curves.

## Setup

```bash
git clone hhttps://github.com/jingjtang/spectral_analysis_web.git
cd spectral_analysis_web

# Create and activate a Python 3.8 environment
python3.8 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run the dashboard (local)

```bash
python main.py
```

Then open the link in your browser (usually http://127.0.0.1:8050/).

## Hosted version
The app is also available on Render: https://spectral-analysis-web.onrender.com
