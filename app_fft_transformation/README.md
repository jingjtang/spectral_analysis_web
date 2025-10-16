# FFT Time Series Explorer

This Dash web application allows you to upload a time series dataset, apply **Fast Fourier Transform (FFT)** analysis, filter specific frequency bands, and visualize both the original and reconstructed signals alongside their frequency-domain spectra.

---

## Features

1. **CSV Upload Support**
   - **Wide schema**: Requires columns `geo_value`, `time_value`, and at least one numeric signal column (e.g., `cases`).
   - **Simple schema**: Requires a date/time column and a numeric column (or just a numeric column — the app will create a synthetic timeline).
   - You can also use the included example file: `piecewise_line_for_testing.csv`.

2. **Signal Selection**
   - Choose a specific **state/region** (`geo_value`) and **signal** (numeric column) to analyze.

3. **Mean Removal (Default On)**
   - If selected, the signal is centered to zero mean before FFT:
     $$
     x_{\text{centered}}(t) = x(t) - \bar{x}
     $$
     where
     $$
     \bar{x} = \frac{1}{N} \sum_{t=0}^{N-1} x(t)
     $$
   - Useful to focus on oscillatory components without DC offset.

4. **Zero Padding**
   - You can pad the series on the left or both sides with zeros before FFT to improve frequency resolution:
     $$
     x_{\text{padded}}(t) =
     \begin{cases}
     0, & t < 0 \\
     x(t), & 0 \le t < N \\
     0, & t \ge N
     \end{cases}
     $$
   - Padding increases the number of FFT bins but **does not** add new information.

5. **Frequency Band Filtering**
   - Apply a **band-pass filter** in the frequency domain:
     - **Hard filter**:
       $$
       X_{\text{filtered}}(f) =
       \begin{cases}
       X(f), & f_{\min} \le |f| \le f_{\max} \\
       0, & \text{otherwise}
       \end{cases}
       $$
     - **Gaussian filter**:
       $$
       X_{\text{filtered}}(f) = X(f) \cdot e^{-\left(\frac{f}{f_{\max}}\right)^2}
       $$

6. **Inverse FFT Reconstruction**
   - After filtering in frequency domain, reconstruct the time-domain signal using inverse FFT:
     $$
     x_{\text{recon}}(t) = \Re\{ \text{IFFT}[X_{\text{filtered}}(f)] \}
     $$

7. **Visualizations**
   - **Time-Domain Plot**:
     - Original signal (gray) — raw or mean-removed, depending on setting.
     - Reconstructed signal (blue) — filtered, with or without mean restoration.
     - Zero-padding is visible as flat 0 segments.
   - **Frequency-Domain Plot**:
     - Amplitude spectrum in the **period domain** (days):
       $$
       A(P) = |X(f)|, \quad P = \frac{1}{f}
       $$
     - Log-log scale for better resolution of both low and high frequency components.

---

## Mathematical Background

### 1. Discrete Fourier Transform (DFT)
For a discrete time series $x[n],\ n=0,1,\dots,N-1$:

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2 \pi k n / N}, \quad k = 0,1,\dots,N-1
$$

Where:
- $X[k]$ is the complex amplitude at frequency $f_k = \frac{k}{N \cdot \Delta t}$.
- $|X[k]|$ gives the amplitude, and $\arg(X[k])$ the phase.

The Fast Fourier Transform (FFT) is an efficient algorithm to compute $X[k]$ in $O(N \log N)$ time.

### 2. Inverse DFT
Reconstruct $x[n]$ from $X[k]$:

$$
x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{i 2 \pi k n / N}
$$

When filtering in the frequency domain, you:
1. Compute $X[k]$ using FFT.
2. Zero out or attenuate unwanted frequencies.
3. Compute the inverse FFT to obtain the filtered time-domain signal.



