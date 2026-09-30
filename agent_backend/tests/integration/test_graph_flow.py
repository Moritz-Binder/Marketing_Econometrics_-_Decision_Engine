import pytest
from unittest.mock import MagicMock
from src.agents.graph import MEDEAgentWorkflow
from src.agents.state import AgentGraphState

@pytest.fixture
def mock_nodes():
    supervisor = MagicMock()
    interceptor = MagicMock()
    auditor = MagicMock()
    synthesizer = MagicMock()
    
    # Just pass state through untouched by default
    supervisor.side_effect = lambda state: state
    interceptor.side_effect = lambda state: state
    auditor.side_effect = lambda state: state
    synthesizer.side_effect = lambda state: state
    
    return supervisor, interceptor, auditor, synthesizer

@pytest.mark.parametrize("intent,expected_called,expected_skipped", [
    ("BUDGET_OPTIMIZATION", "interceptor", "auditor"),
    ("FALLACY_CHECK", "interceptor", "auditor"),
    ("CALIBRATION", "auditor", "interceptor"),
    ("AUDIT_EXPERIMENT", "auditor", "interceptor"),
])
def test_graph_routes_correctly(mock_nodes, intent, expected_called, expected_skipped):
    supervisor, interceptor, auditor, synthesizer = mock_nodes
    node_map = {"interceptor": interceptor, "auditor": auditor}

    supervisor.side_effect = lambda state: {"user_intent": intent}
    workflow = MEDEAgentWorkflow(supervisor, interceptor, auditor, synthesizer)

    workflow.invoke({"raw_user_query": "test query", "messages": []})

    supervisor.assert_called_once()
    synthesizer.assert_called_once()
    node_map[expected_called].assert_called_once()
    node_map[expected_skipped].assert_not_called()

def test_graph_fallback_routes_directly_to_synthesizer(mock_nodes):
    supervisor, interceptor, auditor, synthesizer = mock_nodes
    supervisor.side_effect = lambda state: {"user_intent": "UNKNOWN_INTENT"}

    workflow = MEDEAgentWorkflow(supervisor, interceptor, auditor, synthesizer)
    workflow.invoke({"raw_user_query": "Hello", "messages": []})

    supervisor.assert_called_once()
    interceptor.assert_not_called()
    auditor.assert_not_called()
    synthesizer.assert_called_once()
