import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils.models import predict_tmax_gp, predict_field
from utils.plots import fig_sample_heatmap, fig_tmax_curve
from utils.physics import (
    K_MIN, K_MAX, K_DEFAULT,
    L_MIN, L_MAX, L_DEFAULT,
    Q_MIN, Q_MAX, Q_DEFAULT,
    T_THRESHOLD_DEFAULT, T_SIM, NT,
)


def _render_monte_carlo(k, L, q_max, T_thresh):
    st.markdown("<div class='section-title'>Monte Carlo — Manufacturing Uncertainty</div>",
                unsafe_allow_html=True)
    st.caption("500 GP samples with ±10% variation on k and L (manufacturing tolerances).")

    n_mc = 500
    rng = np.random.default_rng(42)
    k_s = np.clip(k * (1 + rng.normal(0, 0.1, n_mc)), K_MIN, K_MAX)
    L_s = np.clip(L * (1 + rng.normal(0, 0.1, n_mc)), L_MIN, L_MAX)
    T_samples = np.array([predict_tmax_gp(ki, Li, q_max)[0] for ki, Li in zip(k_s, L_s)])

    pct_fail = float(np.mean(T_samples >= T_thresh) * 100)
    p5, p50, p95 = np.percentile(T_samples, [5, 50, 95])

    fig_mc = go.Figure(go.Histogram(x=T_samples, nbinsx=40,
                                     marker_color="#00b4d8", opacity=0.8))
    fig_mc.add_vline(x=T_thresh, line_dash="dash", line_color="#ff6b35",
                     annotation_text=f"Threshold {T_thresh}°C",
                     annotation_font_color="#ff6b35")
    fig_mc.add_vline(x=p50, line_dash="dot", line_color="#90e0ef",
                     annotation_text=f"P50={p50:.0f}°C",
                     annotation_font_color="#90e0ef")
    fig_mc.update_layout(
        template="plotly_dark", paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font=dict(family="JetBrains Mono"),
        xaxis_title="T_max (°C)", yaxis_title="Count",
        title=dict(text="T_max distribution — 500 MC samples", font=dict(color="#00b4d8")),
        margin=dict(l=50, r=30, t=50, b=50),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P5", f"{p5:.0f}°C")
    m2.metric("P50", f"{p50:.0f}°C")
    m3.metric("P95", f"{p95:.0f}°C")
    m4.metric("Failure rate", f"{pct_fail:.1f}%",
              delta="UNSAFE" if pct_fail > 5 else "OK",
              delta_color="inverse")

st.markdown("<h2 style='color:#00b4d8; font-family:Space Grotesk,sans-serif;'>⚡ Live Demo</h2>",
            unsafe_allow_html=True)
st.markdown("<p style='color:#8892a4;'>Real-time GP + MLP prediction with uncertainty quantification</p>",
            unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([1, 1.6], gap="large")

# ── LEFT: Inputs ───────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-title">Input Parameters</div>', unsafe_allow_html=True)

    k = st.slider("Thermal conductivity k (W/m·K)", K_MIN, K_MAX, K_DEFAULT, step=0.1)
    L = st.slider("Thickness L (m)", L_MIN, L_MAX, L_DEFAULT, step=0.002, format="%.3f")
    q_max = st.slider("Max heat flux q_max (W/m²)", Q_MIN, Q_MAX, Q_DEFAULT, step=1000)
    T_thresh = st.slider("Safety threshold T_threshold (°C)", 500, 2500, T_THRESHOLD_DEFAULT, step=50)

    times_arr = np.linspace(0, T_SIM, NT)
    t_idx = st.slider("Display time t (s)", 0, T_SIM, T_SIM // 2, step=10)
    time_index = int(t_idx / T_SIM * (NT - 1))

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.78rem; color:#8892a4; font-family:JetBrains Mono,monospace;'>
    <span style='color:#90e0ef;'>GP</span> → T_max ± σ &nbsp;|&nbsp; 0.001 ms<br>
    <span style='color:#90e0ef;'>MLP</span> → T(x,y,t) &nbsp;|&nbsp; (181,21,21)
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT: Results ─────────────────────────────────────────────────────────
with right:
    st.markdown('<div class="section-title">Predictions</div>', unsafe_allow_html=True)

    T_max, T_std = 0.0, 0.0
    try:
        T_max, T_std = predict_tmax_gp(k, L, q_max)
        go_flag = T_max < T_thresh
        badge_class = "badge-go" if go_flag else "badge-nogo"
        badge_text = "✓ GO" if go_flag else "✗ NO-GO"
        margin = T_thresh - T_max

        rc1, rc2 = st.columns(2)
        rc1.markdown(f"""
        <div class="result-box">
          <div style='font-size:0.7rem; color:#8892a4; text-transform:uppercase; letter-spacing:.09em;'>GP T_max</div>
          <div class="result-value">{T_max:.0f} °C</div>
          <div class="result-sigma">± {T_std:.1f} °C &nbsp;(1σ)</div>
        </div>
        """, unsafe_allow_html=True)
        rc2.markdown(f"""
        <div style='padding:0.5rem 0;'>
          <span class="{badge_class}">{badge_text}</span>
          <div style='font-size:0.78rem; color:#8892a4; margin-top:0.6rem;'>
            Margin: <b style='color:#fafafa;'>{margin:+.0f}°C</b> vs {T_thresh}°C limit
          </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"GP prediction failed: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    try:
        T_field, times = predict_field(k, L, q_max)
        T_slice = T_field[time_index]
        T_curve = np.max(T_field, axis=(1, 2))

        tab1, tab2, tab3 = st.tabs(["🌡 Heatmap T(x,y)", "📈 T_max(t) Curve", "🎲 Monte Carlo"])

        with tab1:
            st.plotly_chart(
                fig_sample_heatmap(T_slice, f"T(x,y) at t = {t_idx} s"),
                use_container_width=True,
            )
            # Export heatmap CSV
            csv_heat = "\n".join(
                [",".join(map(lambda v: f"{v:.2f}", row)) for row in T_slice]
            )
            st.download_button(
                "⬇ Download heatmap CSV",
                data=csv_heat.encode(),
                file_name=f"heatmap_k{k:.1f}_L{L:.3f}_t{t_idx}.csv",
                mime="text/csv",
            )

        with tab2:
            fig_curve = fig_tmax_curve(times, T_curve, T_max, T_std)
            st.plotly_chart(fig_curve, use_container_width=True)
            # Export curve CSV
            csv_curve = "time_s,T_max_C\n" + "\n".join(
                f"{t:.1f},{v:.2f}" for t, v in zip(times, T_curve)
            )
            st.download_button(
                "⬇ Download T_max(t) CSV",
                data=csv_curve.encode(),
                file_name=f"Tmax_curve_k{k:.1f}_L{L:.3f}.csv",
                mime="text/csv",
            )

        with tab3:
            _render_monte_carlo(k, L, q_max, T_thresh)

    except Exception as e:
        st.warning(f"MLP full field not available: {e}")
        st.caption("Heatmap and curve require surrogate_model.pkl")
