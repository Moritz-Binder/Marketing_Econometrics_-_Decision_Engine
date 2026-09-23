import pytest
from unittest.mock import MagicMock, patch
from src.agents.nodes.synthesizer import SynthesizerNode, ExecutiveDecisionBrief, MetricCorrection

@patch("src.agents.nodes.synthesizer.ChatGoogleGenerativeAI")
@patch("src.agents.nodes.synthesizer.get_settings")
def test_synthesizer_node(mock_get_settings, mock_chat_model):
    mock_settings = MagicMock()
    mock_settings.REASONING_MODEL = "gemini-1.5-pro"
    mock_settings.GEMINI_API_KEY = "fake-key"
    mock_get_settings.return_value = mock_settings
    
    mock_structured_llm = MagicMock()
    mock_chat_model.return_value.with_structured_output.return_value = mock_structured_llm
    
    # Define what the LLM should pretend to return
    mock_structured_llm.invoke.return_value = ExecutiveDecisionBrief(
        summary_verdict="Do not scale TV. The channel is saturated.",
        fallacies=[
            MetricCorrection(
                metric="Average vs. Marginal ROAS",
                user_assumption="TV ROAS is 2.5, we should scale it.",
                mathematical_reality="mROAS is 0.8. The channel is over-saturated.",
                risk_level="High",
                capital_at_risk_eur=150000.0
            )
        ],
        recommended_allocations=[],
        methodology_caveats=["Assumes constant macro environment."]
    )
    
    node = SynthesizerNode()
    
    state = {
        "user_intent": "FALLACY_CHECK",
        "raw_user_query": "Let's double TV spend, the ROAS is great.",
        "detected_fallacies": [{"metric": "Average vs. Marginal ROAS", "risk": "High"}],
        "messages": []
    }
    
    update = node(state)
    
    assert "final_brief" in update
    assert update["final_brief"]["summary_verdict"] == "Do not scale TV. The channel is saturated."
    assert len(update["final_brief"]["fallacies"]) == 1
    
    mock_structured_llm.invoke.assert_called_once()
