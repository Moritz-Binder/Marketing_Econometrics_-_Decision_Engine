from typing import Annotated, Any, Dict, List, Optional, TypedDict
from langchain_core.messages import BaseMessage
import operator

class AgentGraphState(TypedDict):
    """
    The core state object passed between nodes in the LangGraph workflow.
    `messages` uses `operator.add` to ensure new messages are appended to the list
    rather than overwriting the existing state.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Query parsing
    raw_user_query: str
    user_intent: str  # e.g., "BUDGET_OPTIMIZATION", "AUDIT_EXPERIMENT", "FALLACY_CHECK", "CALIBRATION"
    target_channels: List[str]
    extracted_parameters: Optional[Dict[str, Any]]
    
    # Intermediate tool outputs
    detected_fallacies: List[Dict[str, Any]]
    raw_query_results: Optional[Dict[str, Any]]
    optimization_results: Optional[Dict[str, Any]]
    audit_results: Optional[Dict[str, Any]]
    calibration_results: Optional[Dict[str, Any]]
    
    # Final output
    final_brief: Optional[Dict[str, Any]]
