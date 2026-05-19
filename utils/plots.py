import numpy as np
import plotly.graph_objects as go
import plotly.express as px

DARK = "plotly_dark"
TEAL = "#00b4d8"
CYAN = "#90e0ef"
ORANGE = "#ff6b35"
CARD_BG = "#1a1f2e"
NAVY = "#0e1117"


def _base_layout(**kwargs):
    return dict(
        template=DARK,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(family="monospace", color="#fafafa"),
        margin=dict(l=50, r=30, t=50, b=50),
        **kwargs,
    )


def fig_sample_heatmap(T_slice, title="Temperature Field T(x,y)"):
    fig = go.Figure(go.Heatmap(
        z=T_slice,
        colorscale="plasma",
        colorbar=dict(title="T (°C)", tickfont=dict(color="#fafafa")),
    ))
    fig.update_layout(
        **_base_layout(title=dict(text=title, font=dict(color=TEAL))),
        xaxis=dict(title="x node", showgrid=False),
        yaxis=dict(title="y node", showgrid=False),
    )
    return fig


def fig_tmax_curve(times, T_curve, T_max_val=None, sigma=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=T_curve,
        mode="lines", name="T_max(t)",
        line=dict(color=TEAL, width=2.5),
    ))
    if T_max_val is not None and sigma is not None:
        peak_idx = int(np.argmax(T_curve))
        fig.add_annotation(
            x=times[peak_idx], y=T_curve[peak_idx],
            text=f"Peak: {T_max_val:.0f} ± {sigma:.0f}°C",
            showarrow=True, arrowhead=2,
            font=dict(color=CYAN, size=11),
            arrowcolor=CYAN,
        )
    fig.update_layout(
        **_base_layout(title=dict(text="T_max over time", font=dict(color=TEAL))),
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="T_max (°C)"),
    )
    return fig


def fig_accuracy_speed():
    data = [
        dict(name="FEM (ref)", time_ms=10_000, error=0.0, marker_size=16, color="#aaaaaa"),
        dict(name="FDM", time_ms=800, error=0.5, marker_size=14, color=ORANGE),
        dict(name="MLP full field", time_ms=1.5, error=9.0, marker_size=14, color="#c084fc"),
        dict(name="MLP scalar", time_ms=0.003, error=0.87, marker_size=14, color=CYAN),
        dict(name="GP scalar", time_ms=0.001, error=0.03, marker_size=18, color=TEAL),
    ]
    fig = go.Figure()
    for d in data:
        fig.add_trace(go.Scatter(
            x=[np.log10(d["time_ms"])],
            y=[d["error"]],
            mode="markers+text",
            name=d["name"],
            text=[d["name"]],
            textposition="top center",
            marker=dict(size=d["marker_size"], color=d["color"],
                        line=dict(width=1.5, color="#ffffff")),
        ))
    fig.update_layout(
        **_base_layout(title=dict(text="Accuracy vs. Speed Trade-off", font=dict(color=TEAL))),
        xaxis=dict(title="log₁₀(Inference Time, ms)", gridcolor="#2a2f3e"),
        yaxis=dict(title="Mean Error (%)", gridcolor="#2a2f3e"),
        showlegend=False,
    )
    return fig


def fig_r2_bars():
    models = ["FDM", "FEM", "MLP full field", "MLP scalar", "GP scalar"]
    r2 = [0.991, 0.9999, 0.9875, 0.9995, 1.0000]
    colors = [ORANGE, "#aaaaaa", "#c084fc", CYAN, TEAL]
    fig = go.Figure(go.Bar(
        x=models, y=r2,
        marker_color=colors,
        text=[f"{v:.4f}" for v in r2],
        textposition="outside",
        textfont=dict(color="#fafafa"),
    ))
    fig.update_layout(
        **_base_layout(title=dict(text="R² Comparison Across Models", font=dict(color=TEAL))),
        yaxis=dict(title="R²", range=[0.96, 1.002], gridcolor="#2a2f3e"),
        xaxis=dict(gridcolor="#2a2f3e"),
    )
    return fig


def fig_design_space(K, L_grid, T_grid, threshold):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=K[0, :], y=L_grid[:, 0], z=T_grid,
        colorscale="plasma",
        colorbar=dict(title="T_max (°C)", tickfont=dict(color="#fafafa")),
        contours=dict(showlabels=True, labelfont=dict(color="#ffffff", size=10)),
        line=dict(width=0.5),
    ))
    # Safety threshold overlay
    fig.add_trace(go.Contour(
        x=K[0, :], y=L_grid[:, 0], z=T_grid,
        contours=dict(
            type="constraint",
            operation="=",
            value=threshold,
            showlabels=True,
            labelfont=dict(color=ORANGE, size=12, family="monospace"),
        ),
        line=dict(color=ORANGE, width=2.5, dash="dash"),
        showscale=False,
        name=f"Threshold {threshold}°C",
    ))
    fig.update_layout(
        **_base_layout(title=dict(
            text=f"Design Space — T_max(k, L) | threshold={threshold}°C",
            font=dict(color=TEAL)
        )),
        xaxis=dict(title="k (W/m·K)", type="log", gridcolor="#2a2f3e"),
        yaxis=dict(title="L (m)", gridcolor="#2a2f3e"),
    )
    return fig


def fig_k_sensitivity(k_vals, T_vals):
    fig = go.Figure(go.Scatter(
        x=k_vals, y=T_vals,
        mode="lines+markers",
        line=dict(color=TEAL, width=2.5),
        marker=dict(color=CYAN, size=5),
    ))
    fig.update_layout(
        **_base_layout(title=dict(text="T_max vs Thermal Conductivity k", font=dict(color=TEAL))),
        xaxis=dict(title="k (W/m·K)", type="log", gridcolor="#2a2f3e"),
        yaxis=dict(title="T_max (°C)", gridcolor="#2a2f3e"),
    )
    return fig
