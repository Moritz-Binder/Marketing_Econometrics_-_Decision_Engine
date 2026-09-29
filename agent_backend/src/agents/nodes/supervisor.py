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
- BUDGET_OPTIMIZATION: The user explicitly asks to allocate budget, run the optimizer, maximize ROI, or calculate optimal spend across channels.
- AUDIT_EXPERIMENT: The user mentions an A/B test, Geo-test, or experiment and discusses stopping it early, sample sizes, p-values, or test validity. ALL questions regarding experiments, A/B testing, or sample sizes MUST be routed here, even if the user is making a flawed claim about the test.
- FALLACY_CHECK: The user is making a dangerous claim about standard marketing metrics, such as pausing a channel due to lack of immediate drop (adstock blindness) or scaling spend purely based on historical Average ROAS (aROAS vs mROAS). If they are misinterpreting a dashboard metric to justify spend changes (not an A/B test), route here.
- CALIBRATION: The user wants to update their MMM priors or bayesian models based on an experimental lift estimate.

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
        
        # Merge target channels and extracted parameters to avoid overwriting API inputs
        final_channels = state.get("target_channels") or []
        if routing_result.target_channels:
            final_channels = list(set(final_channels + routing_result.target_channels))
            
        final_params = state.get("extracted_parameters") or {}
        if routing_result.extracted_parameters:
            final_params.update(routing_result.extracted_parameters)
            
        # Return the partial state update dictionary. 
        # LangGraph will automatically merge these keys into the full AgentGraphState.
        return {
            "user_intent": routing_result.user_intent,
            "target_channels": final_channels,
            "extracted_parameters": final_params,
            "raw_user_query": query
        }

