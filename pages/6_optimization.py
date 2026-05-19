import streamlit as st
import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint
from utils.models import predict_tmax_gp
from utils.plots import fig_design_space
from utils.physics import K_MIN, K_MAX, L_MIN, L_MAX, Q_MIN, Q_MAX, Q_DEFAULT, T_THRESHOLD_DEFAULT

st.markdown("<h2 style='color:#00b4d8; font-family:Space Grotesk,sans-serif;'>🎯 Optimization</h2>",
            unsafe_allow_html=True)
st.markdown("<p style='color:#8892a4;'>Find the minimum-mass TPS design subject to a thermal constraint</p>",
            unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([1, 1.8], gap="large")

with left:
    st.markdown('<div class="section-title">Problem Setup</div>', unsafe_allow_html=True)
    st.latex(r"\min_{k,\, L} \quad \rho \cdot L")
    st.latex(r"\text{s.t.} \quad T_{max}(k, L, q_{max}) \leq T_{threshold}")
    st.latex(r"k \in [0.2,\ 15.0],\quad L \in [0.04,\ 0.12]")

    st.markdown("---")
    st.markdown('<div class="section-title">Controls</div>', unsafe_allow_html=True)

    q_max = st.slider("Heat flux q_max (W/m²)", Q_MIN, Q_MAX, Q_DEFAULT, step=1000, key="opt_q")
    T_thresh = st.slider("Safety threshold (°C)", 300, 2500, T_THRESHOLD_DEFAULT, step=50, key="opt_t")

    run = st.button("▶  Run Optimization", use_container_width=True, type="primary")

    st.markdown("""
    <div style='font-size:0.78rem; color:#8892a4; margin-top:0.8rem; font-family:JetBrains Mono,monospace;'>
    Objective: minimize TPS mass (ρ·L)<br>
    Constraint: T_max ≤ threshold<br>
    Solver: differential_evolution (global)
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">Result</div>', unsafe_allow_html=True)

    if run:
        with st.spinner("Running global optimizer (differential evolution)…"):

            RHO = 1800.0  # kg/m³

            def objective(x):
                k_val, L_val = x
                return RHO * L_val  # minimize mass per unit area

            def constraint_thermal(x):
                k_val, L_val = x
                T, _ = predict_tmax_gp(k_val, L_val, q_max)
                return T_thresh - T  # must be >= 0

            bounds = [(K_MIN, K_MAX), (L_MIN, L_MAX)]

            result = differential_evolution(
                objective,
                bounds,
                constraints=[NonlinearConstraint(constraint_thermal, 0, np.inf)],
                seed=42,
                maxiter=200,
                tol=1e-4,
                workers=1,
                polish=True,
            )

        k_opt, L_opt = result.x
        T_opt, T_std_opt = predict_tmax_gp(k_opt, L_opt, q_max)
        mass_opt = 1800.0 * L_opt
        feasible = T_opt <= T_thresh

        st.markdown(f"""
        <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.2rem;'>
          <div class="opt-card" style='flex:1; min-width:130px;'>
            <div class="opt-value">{k_opt:.2f}</div>
            <div class="opt-label">k* (W/m·K)</div>
          </div>
          <div class="opt-card" style='flex:1; min-width:130px;'>
            <div class="opt-value">{L_opt*100:.2f} cm</div>
            <div class="opt-label">L* (thickness)</div>
          </div>
          <div class="opt-card" style='flex:1; min-width:130px;'>
            <div class="opt-value">{mass_opt:.1f}</div>
            <div class="opt-label">mass* (kg/m²)</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        badge = (
            "<span class='badge-go'>✓ FEASIBLE</span>" if feasible
            else "<span class='badge-nogo'>✗ INFEASIBLE</span>"
        )
        st.markdown(f"""
        {badge}
        <div style='margin-top:0.6rem; font-size:0.85rem; color:#8892a4;'>
          T_max at optimum: <b style='color:#fafafa;'>{T_opt:.1f} °C</b>
          &nbsp;±&nbsp;{T_std_opt:.1f}°C &nbsp;|&nbsp;
          Limit: {T_thresh}°C &nbsp;|&nbsp;
          Margin: <b style='color:#00c864;'>{T_thresh - T_opt:+.1f}°C</b>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Show design space with optimum overlaid
        from utils.models import compute_design_space
        import plotly.graph_objects as go

        with st.spinner("Rendering design space…"):
            K_grid, L_grid, T_grid = compute_design_space(q_max, n=35)

        fig = fig_design_space(K_grid, L_grid, T_grid, T_thresh)
        fig.add_trace(go.Scatter(
            x=[k_opt], y=[L_opt],
            mode="markers",
            marker=dict(symbol="star", size=18, color="#00c864",
                        line=dict(color="#ffffff", width=1.5)),
            name="Optimum",
        ))
        fig.update_layout(showlegend=True,
                          legend=dict(font=dict(color="#fafafa")))
        st.plotly_chart(fig, use_container_width=True)

        # Export
        csv_out = f"k_opt,L_opt,T_max_opt,T_std,mass_kg_m2,feasible\n"
        csv_out += f"{k_opt:.4f},{L_opt:.4f},{T_opt:.2f},{T_std_opt:.2f},{mass_opt:.2f},{feasible}\n"
        st.download_button("⬇ Download result CSV", csv_out.encode(),
                           file_name="tps_optimum.csv", mime="text/csv")

    else:
        st.info("Set the heat flux and safety threshold, then click **Run Optimization**.", icon="🎯")
        st.markdown("""
        The optimizer searches the full (k, L) design space to find the
        thinnest TPS panel (minimum mass) that keeps T_max below the threshold.

        - **k** (thermal conductivity): higher k dissipates heat faster but may
          require exotic materials
        - **L** (thickness): directly proportional to mass and cost
        - **Objective**: minimize ρ·L (areal mass density, kg/m²)
        """)
