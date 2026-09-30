import json
from typing import Dict, Any, List, Literal
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.state import AgentGraphState
from src.core.config import get_settings

# --- API Schemas for Output Structure ---
class MetricCorrection(BaseModel):
    metric: str
    user_assumption: str
    mathematical_reality: str
    risk_level: Literal["Low", "Medium", "High", "Critical"]
    capital_at_risk_eur: float

class ReallocationItem(BaseModel):
    channel: str
    current_spend_eur: float
    recommended_spend_eur: float
    expected_mroas: float
    historical_aroas: float
    saturation_state: Literal["Under-saturated", "Optimal", "Over-saturated"]

class ExecutiveDecisionBrief(BaseModel):
    summary_verdict: str
    fallacies: List[MetricCorrection]
    recommended_allocations: List[ReallocationItem]
    methodology_caveats: List[str]
# ----------------------------------------

class SynthesizerNode:
    def __init__(self) -> None:
        settings = get_settings()
        # Use the powerful reasoning model (gemini-1.5-pro) for final synthesis
        self.llm = ChatGoogleGenerativeAI(
            model=settings.REASONING_MODEL,
            api_key=settings.GEMINI_API_KEY,
            temperature=0.0
        ).with_structured_output(ExecutiveDecisionBrief)
        
        self.system_prompt = """You are a Principal Full-Stack Data Scientist and Executive Communicator.
You receive raw outputs from deterministic econometric engines (Budget Optimizer, Experiment Auditor, Bayesian Calibrator) and Fallacy Checkers.
Your job is to synthesize these findings into a crisp, executive-ready decision brief.

RULES:
1. DO NOT hallucinate any numbers or perform any arithmetic. Use only the exact numbers provided in the tool outputs.
2. If fallacies were detected, emphasize them in the summary verdict and highlight the capital at risk.
3. If this is a BUDGET_OPTIMIZATION, output the recommended allocations provided in the state.
4. If this is an AUDIT or CALIBRATION, frame the verdict around the statistical integrity (SRM, power) or the updated posterior efficiency.
5. Provide strict methodological caveats based on what the tools actually did (e.g., "relies on Hill saturation curve assumptions").
"""

    def __call__(self, state: AgentGraphState) -> Dict[str, Any]:
        intent = state.get("user_intent", "")
        query = state.get("raw_user_query", "")
        
        # Pull all outputs from previous node computations
        context = {
            "user_query": query,
            "intent": intent,
            "detected_fallacies": state.get("detected_fallacies", []),
            "optimization_results": state.get("optimization_results"),
            "audit_results": state.get("audit_results"),
            "calibration_results": state.get("calibration_results")
        }
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Synthesize the following deterministic engine outputs into an Executive Brief:\n\n{json.dumps(context, indent=2)}")
        ]
        
        # Invoke the structured LLM
        brief: ExecutiveDecisionBrief = self.llm.invoke(messages)
        
        return {
            "final_brief": brief.model_dump()
        }
