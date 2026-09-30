import pytest
from unittest.mock import MagicMock, patch
from src.agents.nodes.supervisor import SupervisorNode, SupervisorIntentRouting

@patch("src.agents.nodes.supervisor.ChatGoogleGenerativeAI")
@patch("src.agents.nodes.supervisor.get_settings")
def test_supervisor_node_routing(mock_get_settings, mock_chat_model):
    # Mock settings
    mock_settings = MagicMock()
    mock_settings.ROUTER_MODEL = "gemini-1.5-flash"
    mock_settings.GEMINI_API_KEY = "fake-key"
    mock_get_settings.return_value = mock_settings
    
    # Mock the LLM chain and its structured output response
    mock_structured_llm = MagicMock()
    mock_chat_model.return_value.with_structured_output.return_value = mock_structured_llm
    
    # Define what the LLM should pretend to return
    mock_structured_llm.invoke.return_value = SupervisorIntentRouting(
        user_intent="BUDGET_OPTIMIZATION",
        target_channels=["TV", "TikTok"],
        extracted_parameters={"total_budget": 500000}
    )
    
    # Initialize node
    node = SupervisorNode()
    
    # Create fake state
    initial_state = {
        "raw_user_query": "Reallocate my 500k budget across TV and TikTok to maximize ROI.",
        "messages": []
    }
    
    # Execute node
    state_update = node(initial_state)
    
    # Assert state was updated correctly based on the mocked LLM output
    assert state_update["user_intent"] == "BUDGET_OPTIMIZATION"
    assert state_update["target_channels"] == ["TV", "TikTok"]
    assert state_update["extracted_parameters"] == {"total_budget": 500000}
    assert state_update["raw_user_query"] == "Reallocate my 500k budget across TV and TikTok to maximize ROI."
    
    # Ensure the LLM was actually called
    mock_structured_llm.invoke.assert_called_once()
