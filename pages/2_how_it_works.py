import streamlit as st
import numpy as np
from utils.models import predict_tmax_gp
from utils.plots import fig_k_sensitivity
from utils.physics import L_DEFAULT, Q_DEFAULT

st.markdown("<h2 style='color:#00b4d8;'>How It Works</h2>", unsafe_allow_html=True)
st.markdown("---")

# ── 1. Governing equations ─────────────────────────────────────────────────
st.markdown('<div class="section-title">Governing Equations</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")
with col1:
    st.markdown("**Heat equation (bulk material)**")
    st.latex(r"\rho c_p \frac{\partial T}{\partial t} = k \nabla^2 T")

    st.markdown("**Plasma flux — top face boundary**")
    st.latex(r"-k \frac{\partial T}{\partial n} = q_{max} \sin\left(\frac{\pi t}{t_{entree}}\right)")

with col2:
    st.markdown("**Radiation — bottom face boundary**")
    st.latex(r"-k \frac{\partial T}{\partial n} = \sigma \varepsilon \left(T^4 - T_{stru}^4\right)")

    st.markdown("**Fixed parameters**")
    st.markdown("""
    | Symbol | Value | Unit |
    |--------|-------|------|
    | ρ | 1800 | kg/m³ |
    | cp | 800 | J/(kg·K) |
    | ε | 0.85 | — |
    | Grid | 21 × 21 | nodes |
    | dt | 10 | s |
    """)

st.markdown("---")

# ── 2. FDM vs FEM ──────────────────────────────────────────────────────────
st.markdown('<div class="section-title">FDM vs FEM vs COMSOL Validation</div>', unsafe_allow_html=True)

st.markdown("""
| Method | Solver | R² vs COMSOL | MAE | Notes |
|--------|--------|-------------|-----|-------|
| FDM | Finite Difference | 0.991 | 10.5°C | Fast prototype, structured grid |
| FEM | Finite Element | 0.9999 | 3.3°C | Reference solver, adaptive mesh |
""")

st.markdown("---")

# ── 3. Interactive k slider ────────────────────────────────────────────────
st.markdown('<div class="section-title">Interactive: T_max vs Thermal Conductivity k</div>',
            unsafe_allow_html=True)
st.caption("Drag the slider — GP predicts T_max in real time.")

k_slider = st.slider("k (W/m·K)", 0.2, 15.0, 2.0, step=0.1, key="k_how")

try:
    # Compute curve for sensitivity plot
    k_range = np.logspace(np.log10(0.2), np.log10(15.0), 60)
    T_curve = [predict_tmax_gp(k, L_DEFAULT, Q_DEFAULT)[0] for k in k_range]
    T_current, T_std = predict_tmax_gp(k_slider, L_DEFAULT, Q_DEFAULT)

    fig = fig_k_sensitivity(k_range, T_curve)

    import plotly.graph_objects as go
    fig.add_vline(x=k_slider, line_dash="dash", line_color="#ff6b35",
                  annotation_text=f"k={k_slider:.1f}", annotation_font_color="#ff6b35")
    fig.add_scatter(x=[k_slider], y=[T_current], mode="markers",
                    marker=dict(size=12, color="#ff6b35", symbol="star"),
                    name=f"T_max={T_current:.0f}°C ±{T_std:.0f}")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("T_max", f"{T_current:.1f} °C")
    c2.metric("Uncertainty ±σ", f"{T_std:.1f} °C")

except Exception as e:
    st.warning(f"GP not available: {e}")

st.markdown("---")

# ── 4. Dataset generation ──────────────────────────────────────────────────
st.markdown('<div class="section-title">Dataset Generation</div>', unsafe_allow_html=True)
st.markdown("""
- **500 FEM simulations** with log-uniform sampling of k ∈ [0.2, 15.0] W/(m·K)
- L and q_max sampled uniformly across their design ranges
- Each FEM run: 1800 s simulation, 21×21 grid, 181 time steps
- Output stored: full T(x,y,t) field + scalar T_max
""")

st.markdown("---")

# ── 5. Surrogate comparison ────────────────────────────────────────────────
st.markdown('<div class="section-title">Surrogate Models Comparison</div>', unsafe_allow_html=True)
st.markdown("""
| Model | Output | R² | Mean Error | Speedup | UQ |
|-------|--------|-----|-----------|---------|-----|
| MLP full field | T(x,y,t) — (181,21,21) | 0.9875 | ~9% | 600× | ✗ |
| MLP scalar | T_max | 0.9995 | 0.87% | 300,000× | ✗ |
| **GP scalar** | **T_max ± σ** | **1.0000** | **0.03%** | **100,000×** | **✓** |
""")
st.info("The GP surrogate provides uncertainty quantification (±σ) which is critical "
        "for safety-critical design decisions — no other surrogate offers this.", icon="ℹ️")
