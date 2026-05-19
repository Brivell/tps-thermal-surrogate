import streamlit as st
import numpy as np
from utils.models import compute_design_space
from utils.plots import fig_design_space
from utils.physics import Q_MIN, Q_MAX, Q_DEFAULT, T_THRESHOLD_DEFAULT

st.markdown("<h2 style='color:#00b4d8;'>🗺️ Design Space Explorer</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#8892a4;'>2D contour map of T_max over the (k, L) design space</p>",
            unsafe_allow_html=True)
st.markdown("---")

col_ctrl, col_chart = st.columns([1, 2.2], gap="large")

with col_ctrl:
    st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)

    q_max = st.slider("Heat flux q_max (W/m²)", Q_MIN, Q_MAX, Q_DEFAULT, step=1000,
                      key="ds_q")
    threshold = st.slider("Safety threshold (°C)", 300, 2500, T_THRESHOLD_DEFAULT, step=50,
                           key="ds_thresh")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.82rem; color:#8892a4;'>
    <b style='color:#90e0ef;'>Green region</b>: T_max &lt; threshold (safe)<br>
    <b style='color:#ff6b35;'>Orange dashed</b>: threshold boundary<br>
    <b style='color:#ff6b35;'>Red region</b>: T_max ≥ threshold (unsafe)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Grid: 40×40 = 1,600 GP queries.\n"
        "Cached after first load.",
        icon="⚡",
    )

with col_chart:
    with st.spinner("Computing 40×40 design space grid…"):
        try:
            K, L_grid, T_grid = compute_design_space(q_max, n=40)
            fig = fig_design_space(K, L_grid, T_grid, threshold)
            st.plotly_chart(fig, use_container_width=True)

            # Summary stats
            safe_pct = float(np.mean(T_grid < threshold) * 100)
            T_min_val = float(np.min(T_grid))
            T_max_val = float(np.max(T_grid))

            m1, m2, m3 = st.columns(3)
            m1.metric("Safe designs", f"{safe_pct:.1f}%")
            m2.metric("T_max min", f"{T_min_val:.0f} °C")
            m3.metric("T_max max", f"{T_max_val:.0f} °C")

        except Exception as e:
            st.error(f"Design space computation failed: {e}")
