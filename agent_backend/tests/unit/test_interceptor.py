import pytest
from unittest.mock import MagicMock
import pandas as pd
from src.agents.nodes.interceptor import FallacyInterceptorNode

def test_fallacy_interceptor_node():
    mock_data_engine = MagicMock()
    
    # Mock DuckDB summary: TV spent 15.6m over 156 weeks (100k/week) with an average ROAS of 2.0
    mock_data_engine.get_channel_summary.return_value = pd.DataFrame([
        {"channel": "TV", "total_spend": 15600000.0, "total_revenue": 31200000.0, "aroas": 2.0} 
    ])
    
    mock_optimizer = MagicMock()
    # K = 50k, meaning at 100k weekly spend, we are well past the saturation point.
    mock_optimizer.channel_params = {
        "TV": {"v_max": 200000, "k": 50000, "n": 2, "alpha": 0.7}
    }
    
    node = FallacyInterceptorNode(mock_data_engine, mock_optimizer)
    
    # Test State
    state = {
        "user_intent": "BUDGET_OPTIMIZATION",
        "target_channels": ["TV"],
        "detected_fallacies": [],
        "messages": [],
        "raw_user_query": ""
    }
    
    update = node(state)
    fallacies = update["detected_fallacies"]
    
    # We should have triggered at least 2 fallacies (Average vs Marginal, and Over-saturation)
    assert len(fallacies) >= 2
    
    metrics = [f["metric"] for f in fallacies]
    assert "Average vs. Marginal ROAS" in metrics
    assert "Over-saturation Threshold" in metrics
    
    # Check that Adstock Decay Blindness triggers when the intent is FALLACY_CHECK
    state["user_intent"] = "FALLACY_CHECK"
    state["detected_fallacies"] = []
    
    update_2 = node(state)
    fallacies_2 = update_2["detected_fallacies"]
    metrics_2 = [f["metric"] for f in fallacies_2]
    
    assert "Adstock Decay Blindness" in metrics_2
