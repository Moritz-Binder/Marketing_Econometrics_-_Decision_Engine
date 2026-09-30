import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
from scipy.stats import norm

def render_hill_curves(
    channel_params: Dict[str, Any], 
    current_spends: Dict[str, float], 
    optimized_spends: Dict[str, float]
) -> go.Figure:
    """Plots the non-linear Hill saturation curves for multiple channels."""
    fig = go.Figure()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (channel, params) in enumerate(channel_params.items()):
        v_max = params["v_max"]
        k = params["k"]
        n = params["n"]
        color = colors[i % len(colors)]

        # Generate smooth curve
        max_spend = max(current_spends.get(channel, 0), optimized_spends.get(channel, 0))
        # Draw curve up to 1.5x the max spend for that channel to show saturation asymptote
        x_vals = np.linspace(0, max_spend * 1.5 if max_spend > 0 else 100000, 100)
        y_vals = (v_max * (x_vals ** n)) / ((k ** n) + (x_vals ** n))

        # Add the continuous saturation curve
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines', name=f"{channel} Curve",
            line=dict(color=color, width=2)
        ))

        # Mark current spend
        curr_spend = current_spends.get(channel, 0)
        curr_rev = (v_max * (curr_spend ** n)) / ((k ** n) + (curr_spend ** n))
        fig.add_trace(go.Scatter(
            x=[curr_spend], y=[curr_rev], mode='markers', name=f"{channel} Current",
            marker=dict(color=color, size=10, symbol='circle')
        ))

        # Mark optimized spend
        opt_spend = optimized_spends.get(channel, 0)
        opt_rev = (v_max * (opt_spend ** n)) / ((k ** n) + (opt_spend ** n))
        fig.add_trace(go.Scatter(
            x=[opt_spend], y=[opt_rev], mode='markers', name=f"{channel} Optimized",
            marker=dict(color=color, size=12, symbol='star')
        ))

    fig.update_layout(
        title="Channel Saturation (Hill Curves)",
        xaxis_title="Spend (€)",
        yaxis_title="Incremental Revenue (€)",
        hovermode="x unified",
        template="plotly_white"
    )
    return fig


def render_allocation_comparison(
    current_spends: Dict[str, float], 
    recommended_spends: Dict[str, float]
) -> go.Figure:
    """Side-by-side bar chart showing budget shifts."""
    channels = list(current_spends.keys())
    curr_vals = [current_spends[c] for c in channels]
    rec_vals = [recommended_spends.get(c, 0) for c in channels]

    fig = go.Figure(data=[
        go.Bar(name='Current Spend', x=channels, y=curr_vals, marker_color='#1f77b4'),
        go.Bar(name='Recommended Spend', x=channels, y=rec_vals, marker_color='#2ca02c')
    ])

    fig.update_layout(
        title="Budget Allocation Shift",
        barmode='group',
        xaxis_title="Channel",
        yaxis_title="Spend (€)",
        template="plotly_white"
    )
    return fig


def render_calibration_distribution(
    prior_mean: float, prior_sd: float, 
    post_mean: float, post_sd: float, 
    hdi_95: List[float]
) -> go.Figure:
    """Plots the Bayesian prior vs posterior distributions with 95% HDI."""
    fig = go.Figure()

    # Determine x-axis range based on both distributions (±4 standard deviations)
    min_x = min(prior_mean - 4*prior_sd, post_mean - 4*post_sd)
    max_x = max(prior_mean + 4*prior_sd, post_mean + 4*post_sd)
    x_vals = np.linspace(min_x, max_x, 500)

    # Prior Distribution
    prior_pdf = norm.pdf(x_vals, prior_mean, prior_sd)
    fig.add_trace(go.Scatter(
        x=x_vals, y=prior_pdf, mode='lines', name='Prior (MMM)',
        line=dict(color='gray', dash='dash', width=2)
    ))

    # Posterior Distribution
    post_pdf = norm.pdf(x_vals, post_mean, post_sd)
    fig.add_trace(go.Scatter(
        x=x_vals, y=post_pdf, mode='lines', name='Posterior (Calibrated)',
        line=dict(color='#d62728', width=3),
        fill='tozeroy', fillcolor='rgba(214, 39, 40, 0.2)'
    ))

    # Add vertical lines for 95% HDI
    fig.add_vline(x=hdi_95[0], line_width=1, line_dash="dot", line_color="black")
    fig.add_vline(x=hdi_95[1], line_width=1, line_dash="dot", line_color="black")
    
    # Add HDI shaded region inside the posterior
    fig.add_vrect(
        x0=hdi_95[0], x1=hdi_95[1], 
        fillcolor="rgba(0, 0, 0, 0.1)", opacity=0.5, layer="below", line_width=0,
        annotation_text="95% HDI", annotation_position="top left"
    )

    fig.update_layout(
        title="Bayesian Prior vs. Calibrated Posterior Lift",
        xaxis_title="Effect Size (Lift)",
        yaxis_title="Probability Density",
        template="plotly_white"
    )
    return fig
