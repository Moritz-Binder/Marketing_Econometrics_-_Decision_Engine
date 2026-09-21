from langgraph.graph import StateGraph, START, END
from src.agents.state import AgentGraphState
from src.agents.nodes.supervisor import SupervisorNode
from src.agents.nodes.interceptor import FallacyInterceptorNode
from src.agents.nodes.auditor import CausalAuditorNode
from src.agents.nodes.synthesizer import SynthesizerNode

class MEDEAgentWorkflow:
    def __init__(
        self,
        supervisor: SupervisorNode,
        interceptor: FallacyInterceptorNode,
        auditor: CausalAuditorNode,
        synthesizer: SynthesizerNode
    ):
        self.supervisor = supervisor
        self.interceptor = interceptor
        self.auditor = auditor
        self.synthesizer = synthesizer
        self.graph = self._build_graph()

    def _route_intent(self, state: AgentGraphState) -> str:
        intent = state.get("user_intent")
        
        if intent in ["BUDGET_OPTIMIZATION", "FALLACY_CHECK"]:
            return "interceptor"
        elif intent in ["AUDIT_EXPERIMENT", "CALIBRATION"]:
            return "auditor"
        
        # Safe fallback if classification fails unexpectedly
        return "synthesizer"

    def _build_graph(self):
        builder = StateGraph(AgentGraphState)
        
        # 1. Add all functional nodes to the graph
        builder.add_node("supervisor", self.supervisor)
        builder.add_node("interceptor", self.interceptor)
        builder.add_node("auditor", self.auditor)
        builder.add_node("synthesizer", self.synthesizer)
        
        # 2. Define standard execution edges
        builder.add_edge(START, "supervisor")
        
        # 3. Add conditional routing logic from the supervisor
        builder.add_conditional_edges(
            "supervisor",
            self._route_intent,
            {
                "interceptor": "interceptor",
                "auditor": "auditor",
                "synthesizer": "synthesizer"
            }
        )
        
        # 4. Map the analytical branches to the final synthesizer
        builder.add_edge("interceptor", "synthesizer")
        builder.add_edge("auditor", "synthesizer")
        
        # 5. Terminate
        builder.add_edge("synthesizer", END)
        
        return builder.compile()
        
    def invoke(self, inputs: dict) -> dict:
        """Helper to invoke the compiled langgraph application."""
        return self.graph.invoke(inputs)
