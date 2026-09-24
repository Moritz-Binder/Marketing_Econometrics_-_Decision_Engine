import pytest
import plotly.graph_objects as go
from frontend.components.charts import (
    render_hill_curves,
    render_allocation_comparison,
    render_calibration_distribution
)

def test_render_hill_curves():
    channel_params = {
        "TV": {"v_max": 1000000, "k": 50000, "n": 1.2},
        "Google": {"v_max": 500000, "k": 20000, "n": 1.5}
    }
    current_spends = {"TV": 40000, "Google": 15000}
    optimized_spends = {"TV": 60000, "Google": 25000}

    fig = render_hill_curves(channel_params, current_spends, optimized_spends)
    
    assert isinstance(fig, go.Figure)
    # 3 traces per channel: the curve line, the current point, the optimized point
    assert len(fig.data) == 6
    assert fig.layout.title.text == "Channel Saturation (Hill Curves)"

def test_render_allocation_comparison():
    current_spends = {"TV": 40000, "Google": 15000}
    recommended_spends = {"TV": 60000, "Google": 25000}

    fig = render_allocation_comparison(current_spends, recommended_spends)
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 # Two bar groups
    assert fig.layout.barmode == 'group'

def test_render_calibration_distribution():
    fig = render_calibration_distribution(
        prior_mean=1.5,
        prior_sd=0.2,
        post_mean=1.8,
        post_sd=0.1,
        hdi_95=[1.6, 2.0]
    )
    
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2 # Prior and posterior traces
    assert fig.layout.title.text == "Bayesian Prior vs. Calibrated Posterior Lift"
