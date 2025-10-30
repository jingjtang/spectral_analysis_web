# COVID Delay-Aware Deconvolution Dashboard

This Dash app provides an interactive environment to **reconstruct infection curves** from confirmed COVID-19 cases, using **delay distributions** between infection and case reporting. It supports both **synthetic Gamma-based delays** and **real-world line-list data**.  

The app also visualizes delay kernels, fits simple Gamma distributions (method of moments), and provides frequency-domain insights via the FFT power spectrum.  

Additionally, it allows injection of high-frequency signals into latent infection curves to demonstrate how very different infection dynamics can still produce similar observed case curves after applying a delay distribution.

---

## Features

- **Upload-free default data**: Uses U.S. state-level confirmed cases (`combined_state_no_revision.csv`).
- **Delay modeling options**:
  - *Synthetic Gamma distribution*: parameterized by mean and scale.  
  - *Real delay distributions* from published linelist datasets:
    - China (30 provinces included, [Zhang et al. (2020)](https://www.thelancet.com/journals/laninf/article/PIIS1473-3099(20)30230-9/fulltext))
    - US CDC ([COVID-19 Case Surveillance line list](https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Restricted-Access-Detai/mbd7-r32t/about_data))
    - Hong Kong ([Chen et al. (2025)](https://www.nature.com/articles/s41467-025-60591-x))
- **Rolling kernel estimation**:
  - Histogram-based  
  - Weighted Kernel Density Estimation (KDE)  
- **Deconvolution methods**:
  - FFT deconvolution  
  - Wiener deconvolution
- **High-frequency signal injection**:
  -	Add synthetic sine wave components to latent infections before reconvolution
	-	Control both amplitude and frequency of injected signal
	-	Demonstrate indistinguishability in confirmed cases due to delay smoothing
- **Diagnostics**:
  - Forward simulation (reconvolution) of reconstructed infections to compare with observed cases  
  - RMSE error metrics for fit quality
- **Visualizations**:
  - Observed vs. reconstructed vs. reconvolved case curves  
  - Delay kernel distribution with Gamma fit summary  
  - FFT power spectrum of the delay distribution  

---

## How It Works

### 1. Infection Reconstruction (Figure 1)

We observe confirmed cases \( C(t) \), which are modeled as the **convolution** of the latent infections \( I(t) \) with a delay distribution (kernel) \( g(\tau) \):  

\[
C(t) \;=\; (I * g)(t) \;=\; \sum_{\tau=0}^{\infty} I(t-\tau) \, g(\tau).
\]

The app provides two methods to **deconvolve** the observed curve \( C(t) \) and recover an estimate of the infection curve \( \hat{I}(t) \):  

- **FFT deconvolution**  
- **Wiener deconvolution** (regularized, to stabilize against noise)

After deconvolution, we perform a **forward check** by reconvolving \( \hat{I}(t) \) with the same kernel \( g(\tau) \):  

$$
\hat{C}(t) = (\hat{I} * g)(t).
$$

This reconvolved curve \( \hat{C}(t) \) is plotted against the observed \( C(t) \) to verify the accuracy of reconstruction.  
Error metrics (e.g. RMSE) are computed to quantify the fit.  

---

### 2. Delay Distribution Visualization (Figure 2)

The second panel focuses **only** on the delay distribution itself.  

- When using **synthetic delays**, the kernel is parameterized as a Gamma distribution:

$$
g(\tau; k, \theta) = \frac{1}{\Gamma(k)\,\theta^k} \tau^{k-1} e^{-\tau/\theta}, \quad \tau \geq 0
$$

with shape \( k \) and scale \( \theta \).  

- When using **real linelist data**, the empirical delay distribution is estimated either by:
  - Histogram of observed delays  
  - Weighted kernel density estimation (KDE)  

Because real delays may vary over time, the app allows you to **select an end window date** and compute the distribution from data up to that point.  
Alternatively, you can manually increase the window size to approximate the **full distribution across all cases**.  


---

### 3. High-Frequency Signal Injection (New Feature)

To test identifiability limits, the app allows the addition of a synthetic high-frequency component to the reconstructed latent infections $\hat{I}(t)$:

$$
\hat{I}_{\text{alt}}(t) = \hat{I}(t) + A \cdot \sin(2\pi f t)
$$

where:

- $A$: user-selected amplitude  
- $f$: user-selected frequency in cycles per day  

This modified infection curve is then **reconvolved** with the delay kernel $g(\\tau)$:

$$
\hat{C}_{\text{alt}}(t) = (\hat{I}_{\text{alt}} * g)(t)
$$

If $\\hat{C}_{\\text{alt}}(t) \\approx \\hat{C}(t) \\approx C(t)$, this demonstrates that **high-frequency patterns in infection curves may be smoothed out and become indistinguishable** in the observed data due to the effect of the delay distribution.

Both parameters $A$ and $f$ are adjustable via the control panel on the left side of the app.

---
Together, these two parts let you:
1. Recover latent infection dynamics from reported cases (with verification).  
2. Inspect and analyze the delay distribution structure itself, including its evolution over time.
3. Experiment with frequency-based signal components to test non-identifiability due to delay filtering.




