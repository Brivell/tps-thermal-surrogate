# TPS Thermal Surrogate — Physics-Based ML for Spacecraft Reentry

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A high-fidelity surrogate model for thermal protection system (TPS) simulation during spacecraft atmospheric reentry. Combines FDM, FEM numerical solvers with Gaussian Process and MLP surrogate models — achieving **R²=1.000** and **0.03% error** at **100,000× speedup** over full FEM simulation.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Governing Equations](#governing-equations)
- [Numerical Methods](#numerical-methods)
  - [Finite Difference Method (FDM)](#finite-difference-method-fdm)
  - [Finite Element Method (FEM)](#finite-element-method-fem)
- [Dataset Generation](#dataset-generation)
- [Surrogate Models](#surrogate-models)
  - [MLP — Full Field Surrogate](#mlp--full-field-surrogate)
  - [MLP Scalar Surrogate](#mlp-scalar-surrogate)
  - [Gaussian Process Surrogate](#gaussian-process-surrogate)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)

---

## Problem Statement

During atmospheric reentry, a spacecraft's thermal protection system (TPS) is subjected to an intense plasma heat flux. The structure must withstand temperatures exceeding 1500°C without melting. The governing challenge is:

> **Given material properties (k), geometry (L), and heat flux (q_max), predict the maximum temperature T_max reached by the structure.**

Full FEM simulation takes ~90 seconds per evaluation — prohibitive for design optimization. This project builds surrogate models that predict T_max **instantaneously** with engineering-grade accuracy.

---

## Governing Equations

### 2D Transient Heat Equation

The temperature field T(x, y, t) satisfies:

$$\rho c_p \frac{\partial T}{\partial t} = k \left( \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} \right)$$

with the following boundary conditions:

**Top face (plasma flux, Neumann BC):**

$$-k \frac{\partial T}{\partial n} = q_{plasma}(t)$$

where the plasma flux follows a sinusoidal profile:

$$q_{plasma}(t) = q_{max} \sin\left(\frac{\pi t}{t_{entree}}\right), \quad 0 \leq t \leq t_{entree}$$

**Bottom face (radiation, nonlinear Robin BC):**

$$-k \frac{\partial T}{\partial n} = \sigma \varepsilon \left( T^4 - T_{stru}^4 \right)$$

**Parameters:**

| Symbol | Description | Value |
|--------|-------------|-------|
| ρ | Density | 1800 kg/m³ |
| c_p | Specific heat | 800 J/(kg·K) |
| k | Thermal conductivity | 0.2–15 W/(m·K) |
| ε | Emissivity | 0.85 |
| σ | Stefan-Boltzmann constant | 5.67×10⁻⁸ W/(m²·K⁴) |
| t_entree | Plasma entry duration | 1200 s |
| T_initial | Initial temperature | 293.15 K |

---

## Numerical Methods

### Finite Difference Method (FDM)

The FDM discretizes the domain on a uniform 21×21 grid with grid spacing Δx = Δy = L/20. Using an implicit Euler scheme with time step Δt = 10s:

$$\rho c_p \frac{T_{i,j}^{n+1} - T_{i,j}^n}{\Delta t} = k \frac{T_{i-1,j}^{n+1} - 2T_{i,j}^{n+1} + T_{i+1,j}^{n+1}}{\Delta x^2} + k \frac{T_{i,j-1}^{n+1} - 2T_{i,j}^{n+1} + T_{i,j+1}^{n+1}}{\Delta y^2}$$

The radiation boundary condition is linearized as:

$$h_{rad}(T) = \sigma \varepsilon (T^2 + T_{stru}^2)(T + T_{stru})$$

**Validation vs COMSOL:** R² = 0.991, MAE = 10.5°C

---

### Finite Element Method (FEM)

The FEM uses **bilinear quadrilateral Q4 elements** with **2×2 Gauss integration**. The weak form of the heat equation leads to the discrete system:

$$\left(\frac{\mathbf{M}}{\Delta t} + \mathbf{K} + \mathbf{K}_{rad}\right) \mathbf{T}^{n+1} = \frac{\mathbf{M}}{\Delta t} \mathbf{T}^n + \mathbf{f}_{plasma} + \mathbf{R}_{rad}$$

**Element matrices** assembled via Gauss quadrature (ξ, η ∈ {-1/√3, 1/√3}):

$$K_e = \int_{\Omega_e} k \left( \nabla \mathbf{N}^T \nabla \mathbf{N} \right) d\Omega \approx \sum_{gp} k \left( \mathbf{B}^T \mathbf{B} \right) |J| w_i w_j$$

$$M_e = \int_{\Omega_e} \rho c_p \mathbf{N}^T \mathbf{N} \, d\Omega$$

**Radiation — Newton-Raphson linearization:**

The nonlinear radiation term σεT⁴ is handled via Newton-Raphson iterations. The tangent stiffness matrix and residual vector on the bottom boundary are:

$$K_{rad}^e = \int_{\Gamma} 4\sigma\varepsilon T^3 \psi_i \psi_j \, d\Gamma$$

$$R_{rad}^e = \int_{\Gamma} \sigma\varepsilon \left(3T^4 + T_{stru}^4\right) \psi_i \, d\Gamma$$

**Adaptive under-relaxation** prevents divergence for low-conductivity materials:

| k (W/m·K) | Relaxation ω |
|-----------|-------------|
| k < 0.5   | 0.2 |
| k < 1.0   | 0.4 |
| k ≥ 1.0   | 0.7 |

**Plasma flux** applied with Crank-Nicolson scheme:

$$\mathbf{f}_{plasma} = \frac{q(t^n) + q(t^{n+1})}{2} \int_{\Gamma_{top}} \mathbf{N}^T \, d\Gamma$$

**Validation vs COMSOL:** R² = 0.9999, MAE = 3°C ✅

---

## Dataset Generation

### Parameter Space

| Parameter | Range | Sampling |
|-----------|-------|----------|
| k (W/m·K) | [0.2, 15.0] | **Log-uniform** |
| L (m) | [0.04, 0.12] | Uniform |
| q_max (W/m²) | [30,000, 80,000] | Uniform |

### Why Log-Uniform Sampling for k?

The thermal conductivity k spans two orders of magnitude (0.2 → 15 W/m·K), and the temperature response T_max varies **exponentially** with k:

$$T_{max} \propto \exp(-\alpha k)$$

With uniform sampling, only ~6/500 samples fall in the critical k < 0.5 region — where temperatures are highest and the problem is most nonlinear. This causes surrogate models to perform poorly on extreme cases.

**Log-uniform sampling** ensures equal coverage across all decades:

```python
k_samples = np.exp(
    np.random.uniform(np.log(0.2), np.log(15.0), n_samples)
)
```

**Result:**

| k range | Uniform (500 samples) | Log-uniform (500 samples) |
|---------|----------------------|--------------------------|
| k < 0.5 | ~6 samples ❌ | ~114 samples ✅ |
| k < 1.0 | ~26 samples | ~193 samples |
| k > 5.0 | ~333 samples | ~133 samples |

### Generation Process

Each sample runs a full FEM simulation (dt=10s, 180 time steps, 21×21 mesh) and stores the complete temperature field T(x,y,t) ∈ ℝ^(181×21×21). A cache system saves each simulation individually to allow automatic resume on crash.

**500 simulations generated in ~52 min on 8-core Linux VM (Hetzner CCX33).**

---

## Surrogate Models

### MLP — Full Field Surrogate

Predicts the **complete temperature field** T(x,y,t) at all spatial points and time steps.

**Architecture:**

```
Input  : (k, L, q_max)  →  3 neurons
Hidden : (1024, 512, 256, 128)  →  ReLU activation
Output : T(x,y,t) flattened  →  181 × 21 × 21 = 79,821 neurons
```

**Training:** Adam optimizer, early stopping (patience=15), StandardScaler normalization.

| Metric | Value |
|--------|-------|
| R² test | 0.9875 |
| MAE | 8.3°C |
| Speedup vs FEM | 600× |

**Limitation:** The MLP predicts 79,821 output values simultaneously from only 3 inputs — a highly under-constrained problem. This leads to regression-to-the-mean behavior on extreme cases (k < 0.5), resulting in ~9% error on T_max benchmark.

---

### MLP Scalar Surrogate

Predicts only **T_max global** (one scalar value) with log-transform on both input k and output T_max.

**Why log-transform?**

The relationship T_max(k) follows an exponential decay:

$$T_{max}(k) \approx A \cdot e^{-\alpha k}$$

Applying log-transform linearizes this relationship:

$$\log(T_{max}) \approx \log(A) - \alpha k$$

This makes the regression problem much easier for the MLP.

```python
X_log = [log(k), L, q_max]
y_log = log(T_max)
```

| Metric | MLP (no log) | MLP (with log) |
|--------|-------------|----------------|
| R² test | 0.892 | 0.9995 |
| MAE | 58.6°C | 4.7°C |
| Erreur moy | 8.67% | 0.87% |

---

### Gaussian Process Surrogate

**Why GP instead of MLP for scalar prediction?**

With only 500 training samples, the GP **outperforms** the MLP fundamentally:

| Aspect | MLP | GP |
|--------|-----|-----|
| Mechanism | Learns fixed weights | Bayesian interpolation between training points |
| With 500 samples | Underfitting risk | Ideal regime |
| Extrapolation | Regresses to mean | Uncertainty increases (honest) |
| Uncertainty | ❌ None | ✅ Quantified ±σ |
| Training complexity | O(n) | O(n³) → limited to ~5000 samples |

The GP with **Matérn ν=2.5 kernel** is particularly well-suited for physical problems because it assumes the underlying function is twice differentiable — consistent with heat equation solutions.

**Kernel:**

$$k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{5}d}{\ell} + \frac{5d^2}{3\ell^2}\right) \exp\left(-\frac{\sqrt{5}d}{\ell}\right)$$

where d is the normalized distance between points and ℓ is the learned length scale.

**The GP also provides prediction uncertainty:**

```python
T_max, T_std = gp.predict(X_new, return_std=True)
# → T_max = 1565°C ± 1°C  (2σ confidence interval)
```

This is critical for engineering decisions — the designer knows not just the predicted temperature but also the confidence in that prediction.

| Metric | MLP Scalar | GP |
|--------|-----------|-----|
| R² test | 0.9995 | **1.0000** |
| MAE | 4.7°C | **0.31°C** |
| Error mean | 0.87% | **0.03%** |
| Error max | 2.83% | **0.63%** |
| Points < 2% | 53/60 | **60/60** |
| Uncertainty | ❌ | ✅ ±1°C |
| Speedup vs FEM | ~300,000× | ~100,000× |

---

## Results Summary

### Validation vs COMSOL

| Method | R² | MAE | Error mean |
|--------|-----|-----|-----------|
| FDM | 0.991 | 10.5°C | 1.53% |
| **FEM** | **0.9999** | **3.3°C** | **0.82%** |

### Surrogate Benchmark (vs FEM)

| Surrogate | Output | R² | Error | Speedup |
|-----------|--------|-----|-------|---------|
| MLP full field | T(x,y,t) | 0.9875 | ~9% | 600× |
| MLP scalar | T_max | 0.9995 | 0.87% | 300,000× |
| **GP scalar** | **T_max ± σ** | **1.0000** | **0.03%** | **100,000×** |

### GP Benchmark by k value

| Case | FEM | GP | ±σ | Error | Speedup |
|------|-----|-----|-----|-------|---------|
| k=0.3 (very low) | 1616°C | 1616°C | ±1°C | 0.01% | 293,073× |
| k=0.5 (low) | 1565°C | 1566°C | ±1°C | 0.03% | 153,848× |
| k=2.0 (medium) | 948°C | 948°C | ±1°C | 0.00% | 81,681× |
| k=8.0 (high) | 574°C | 573°C | ±0°C | 0.02% | 92,969× |
| k=14.0 (very high) | 370°C | 371°C | ±1°C | 0.24% | 71,266× |

---

## Installation

```bash
git clone https://github.com/[username]/tps-thermal-surrogate
cd tps-thermal-surrogate
pip install -r requirements.txt
```

---

## Usage

### Run FEM simulation

```python
from tps_fct_fem import simulation_principale
import numpy as np

class prm:
    rho=1800; cp=800; k_therm=2.0; epsilon=0.85

Res = simulation_principale(
    [0, 0.1], [0, 0.1], 21, 21, 10.0, 1800.0,
    temps_instantanes=[], T_initiale=293.15, T_stru=293.15,
    t_entree=1200.0, q_max=50000, sigma=5.67e-8, prm=prm,
    save_all=True, verbose=True
)
print(f"T_max = {np.max(Res['T_field_complet']):.1f}°C")
```

### Predict with GP surrogate

```python
import pickle, numpy as np

with open('surrogate_gp.pkl', 'rb') as f:
    gp, scaler_X, scaler_y = pickle.load(f)

k, L, q_max = 2.0, 0.08, 50000
X_new = scaler_X.transform([[np.log(k), L, q_max]])
T_log, T_std = gp.predict(X_new, return_std=True)
T_max = np.exp(scaler_y.inverse_transform(T_log.reshape(-1,1)))[0][0]
T_unc = T_max * T_std[0] * scaler_y.scale_[0]

print(f"T_max = {T_max:.1f}°C ± {T_unc:.1f}°C")
```

### Generate dataset

```python
from ml_surrogate1 import generate_dataset
generate_dataset(n_samples=500, save_file='dataset_TPS.npz')
```

---

## Project Structure

```
tps-thermal-surrogate/
├── tps_fct_fdm.py              # FDM solver
├── tps_fct_fem.py              # FEM solver (Q4 elements, Newton-Raphson)
├── ml_surrogate1.py            # Full field MLP surrogate
├── train_surrogate_scalar.py   # Scalar MLP surrogate with log-transform
├── train_surrogate_gp.py       # Gaussian Process surrogate
├── validate_comsol.py          # Validation vs COMSOL
├── dataset_TPS.npz             # 500 FEM simulations (log-uniform sampling)
├── surrogate_model.pkl         # Trained full field MLP
├── surrogate_gp.pkl            # Trained GP model
├── surrogate_tmax_scalar.pkl   # Trained scalar MLP
├── requirements.txt
└── README.md
```

---

## Future Work

- [ ] **1000 samples** with log-uniform sampling → expected R² > 0.99 for full field
- [ ] **CNN decoder** — exploit spatial structure (21×21) → expected error ~2-3%
- [ ] **DeepONet** — operator learning for T(x,y,t) prediction → expected error ~0.5-1%
- [ ] **PINN** — physics-informed neural network (no FEM data needed) → expected error ~0.1%
- [ ] **Streamlit app** — interactive design tool with GP uncertainty visualization
- [ ] **Design optimization** — find optimal (k, L) for T_max < threshold using GP

---

## References

- De Farias & Van Roy (2003) — Approximate Linear Programming
- Zienkiewicz & Taylor — The Finite Element Method
- Rasmussen & Williams (2006) — Gaussian Processes for Machine Learning
- NASA TPS design guidelines

---

## License

MIT License — see [LICENSE](LICENSE) for details.
