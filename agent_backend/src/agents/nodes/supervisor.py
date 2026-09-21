from typing import List, Optional, Any, Dict, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.state import AgentGraphState
from src.core.config import get_settings

class SupervisorIntentRouting(BaseModel):
    """Structured output for the Supervisor Node to classify intent and extract entities."""
    user_intent: Literal["BUDGET_OPTIMIZATION", "AUDIT_EXPERIMENT", "FALLACY_CHECK", "CALIBRATION", "UNKNOWN"] = Field(
        ..., 
        description="The primary intent of the user's query."
    )
    target_channels: List[str] = Field(
        default_factory=list, 
        description="List of marketing channels mentioned (e.g., 'TV', 'Google_Search', 'Meta_Display', 'TikTok', 'YouTube')."
    )
    extracted_parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Any specific numbers, budgets, or metrics mentioned in the query (e.g., total_budget, control_traffic)."
    )

class SupervisorNode:
    def __init__(self) -> None:
        settings = get_settings()
        # Initialize the router model (Gemini 1.5 Flash) with temperature=0 for deterministic routing
        self.llm = ChatGoogleGenerativeAI(
            model=settings.ROUTER_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=0.0
        ).with_structured_output(SupervisorIntentRouting)
        
        self.system_prompt = """You are an elite Principal Marketing Scientist and AI Platform Router.
Your job is to analyze a senior marketing stakeholder's query and route it to the correct deterministic analytical engine.

Intents:
- BUDGET_OPTIMIZATION: The user wants to allocate budget, maximize ROI, or evaluate the impact of shifting spend.
- AUDIT_EXPERIMENT: The user has run an A/B test or Geo-experiment and wants to audit it for SRM, power, or validity.
- FALLACY_CHECK: The user is making a high-risk claim (e.g. cutting a channel completely) and needs a rigorous fallacy check on their reasoning.
- CALIBRATION: The user wants to update their MMM priors based on an experimental lift estimate.

Extract the intent and any specific channels mentioned."""

    def __call__(self, state: AgentGraphState) -> Dict[str, Any]:
        """
        Executes the node. Reads the query, invokes the LLM, and returns the state updates.
        """
        query = state.get("raw_user_query", "")
        if not query and state.get("messages"):
            query = str(state["messages"][-1].content)
            
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query)
        ]
        
        # Invoke the structured LLM
        routing_result: SupervisorIntentRouting = self.llm.invoke(messages)
        
        # Return the partial state update dictionary. 
        # LangGraph will automatically merge these keys into the full AgentGraphState.
        return {
            "user_intent": routing_result.user_intent,
            "target_channels": routing_result.target_channels,
            "extracted_parameters": routing_result.extracted_parameters,
            "raw_user_query": query
        }

