import streamlit as st
from utils.plots import fig_accuracy_speed, fig_r2_bars

st.markdown("<h2 style='color:#00b4d8;'>Benchmarks</h2>", unsafe_allow_html=True)
st.markdown("---")

# ── Accuracy vs Speed ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">Accuracy vs. Speed Trade-off</div>', unsafe_allow_html=True)
st.plotly_chart(fig_accuracy_speed(), use_container_width=True)
st.caption(
    "x-axis: log₁₀(inference time in ms). "
    "GP achieves near-zero error at 0.001 ms — best of both dimensions."
)

st.markdown("---")

# ── GP Benchmark table ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">GP Benchmark vs FEM by k</div>', unsafe_allow_html=True)

st.markdown("""
| k (W/m·K) | FEM T_max | GP T_max | ±σ | Error | Speedup |
|-----------|----------|---------|-----|-------|---------|
| 0.3 | 1616 °C | 1616 °C | ±1°C | 0.01% | 293,073× |
| 0.5 | 1565 °C | 1566 °C | ±1°C | 0.03% | 153,848× |
| 2.0 | 948 °C | 948 °C | ±1°C | 0.00% | 81,681× |
| 8.0 | 574 °C | 573 °C | ±0°C | 0.02% | 92,969× |
| 14.0 | 370 °C | 371 °C | ±1°C | 0.24% | 71,266× |
""")

col1, col2, col3 = st.columns(3)
col1.metric("Best Speedup", "293,073×", "k=0.3 W/m·K")
col2.metric("Best Error", "0.00%", "k=2.0 W/m·K")
col3.metric("Avg ±σ", "~±1°C", "across all k")

st.markdown("---")

# ── R² Bar chart ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">R² Comparison Across Models</div>', unsafe_allow_html=True)
st.plotly_chart(fig_r2_bars(), use_container_width=True)

st.markdown("---")

# ── UQ note ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Uncertainty Quantification — GP Advantage</div>',
            unsafe_allow_html=True)
st.markdown("""
The Gaussian Process not only matches FEM accuracy (R²=1.0000) but also
**quantifies prediction uncertainty** — a feature absent in neural network surrogates.

This enables:
- **Reliability-based design**: reject designs where GP uncertainty exceeds a safety margin
- **Active learning**: identify regions of parameter space where more FEM data is needed
- **Risk-aware optimization**: penalize high-σ designs in MDO loops

For spacecraft TPS, operating at **0.001 ms per query**, this makes Monte Carlo
uncertainty propagation (10,000+ samples) computationally trivial.
""")
