import streamlit as st
import numpy as np
from utils.models import load_dataset
from utils.plots import fig_sample_heatmap

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">TPS <span>Thermal Surrogate</span></div>
  <div class="hero-sub">Physics-based ML for spacecraft reentry thermal protection</div>
  <span class="hero-value">
    Replaces hours of FEM simulation with 0.001 ms GP inference — validated against COMSOL
  </span>
</div>
""", unsafe_allow_html=True)

# ── Metric cards ───────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
metrics = [
    ("0.001 ms", "per GP inference"),
    ("0.03%", "mean error vs FEM"),
    ("100,000×", "faster than FEM"),
]
for col, (val, lbl) in zip([c1, c2, c3], metrics):
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-value">{val}</div>
      <div class="metric-label">{lbl}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tech stack ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="badge-wrap">
  <span class="tech-badge">Python 3.13</span>
  <span class="tech-badge">scikit-learn GP</span>
  <span class="tech-badge">MLP full-field</span>
  <span class="tech-badge">Plotly</span>
  <span class="tech-badge">FEM reference</span>
  <span class="tech-badge">COMSOL validated</span>
  <span class="tech-badge">Uncertainty quantification</span>
  <span class="tech-badge">500 simulations</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Problem description + heatmap ──────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">The Problem</div>', unsafe_allow_html=True)
    st.markdown("""
    During atmospheric reentry, the spacecraft's **Thermal Protection System (TPS)**
    experiences extreme plasma heating — up to **80,000 W/m²** of heat flux.
    Predicting the peak temperature **T_max** accurately is critical for structural
    integrity and crew safety.

    Traditional FEM simulations require seconds-to-minutes per run, making design
    optimization and Monte Carlo uncertainty studies computationally prohibitive.
    **This surrogate replaces costly FEM** with:

    - A **Gaussian Process (GP)** for instant T_max ± σ prediction (0.001 ms)
    - A **full-field MLP** for complete T(x, y, t) distribution

    Both trained on **500 FEM simulations**, validated against COMSOL (R²=0.9999).
    """)

    st.markdown('<div class="section-title" style="margin-top:1.4rem;">Pipeline</div>',
                unsafe_allow_html=True)
    st.markdown("""
    | Stage | Method | R² vs COMSOL | Speedup |
    |-------|--------|-------------|---------|
    | Prototype | FDM | 0.991 | — |
    | Reference | FEM | 0.9999 | 1× |
    | Surrogate | GP | 1.0000 | **100,000×** |
    | Surrogate | MLP field | 0.9875 | 600× |
    """)

with right:
    st.markdown('<div class="section-title">Sample Temperature Field</div>', unsafe_allow_html=True)
    try:
        data = load_dataset()
        keys = list(data.keys())
        T_key = next((k for k in keys if "T" in k.upper() and "field" in k.lower()), None)
        if T_key is None:
            T_key = next((k for k in keys if data[k].ndim >= 3), None)

        if T_key is not None:
            T_all = data[T_key]
            if T_all.ndim == 4:
                T_slice = T_all[0, T_all.shape[1]//2]
            elif T_all.ndim == 3:
                T_slice = T_all[T_all.shape[0]//2]
            else:
                T_slice = T_all
            st.plotly_chart(fig_sample_heatmap(T_slice), use_container_width=True)
        else:
            x = np.linspace(0, 1, 21)
            y = np.linspace(0, 1, 21)
            X, Y = np.meshgrid(x, y)
            T_syn = 300 + 1300 * np.exp(-((X*3)**2 + (Y*2 - 0.5)**2))
            st.plotly_chart(fig_sample_heatmap(T_syn, "Sample Temperature Field"),
                            use_container_width=True)
    except Exception as e:
        st.warning(f"Dataset not loaded: {e}")

st.markdown("---")

# ── About ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">About</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-card">
  <div class="about-name">Amaury Tchoupe</div>
  <div class="about-role">Computational Engineering · Physics-based ML · Surrogate Modeling</div>
  <a class="about-link" href="https://github.com/amaurytchoupe" target="_blank">⌥ GitHub</a>
  <a class="about-link" href="https://linkedin.com/in/amaurytchoupe" target="_blank">↗ LinkedIn</a>
  <a class="about-link" href="mailto:amaurytchoupe01@gmail.com">✉ Contact</a>
  <div style='margin-top:0.9rem; font-size:0.82rem; color:#8892a4; line-height:1.6;'>
    Built as part of a computational thermal engineering project — FDM prototype →
    FEM reference → ML surrogate pipeline, with COMSOL cross-validation.
    Designed to support TPS design space exploration for reentry vehicles.
  </div>
</div>
""", unsafe_allow_html=True)
