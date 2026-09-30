import streamlit as st
import json
from typing import Dict, Any

def render_trace_viewer(raw_state: Dict[str, Any]):
    """
    Renders a collapsible debugging view showing the agent's internal state,
    intent routing, and raw tool outputs.
    """
    if not raw_state:
        return

    with st.expander("🔍 Agent Reasoning & Econometric Trace"):
        st.markdown("""
        *This panel provides transparency into the deterministic tools and LLM routing 
        decisions executed under the hood by the MEDE LangGraph workflow.*
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**LLM Supervisor Routing**")
            intent = raw_state.get("user_intent", "UNKNOWN")
            channels = raw_state.get("target_channels", [])
            st.info(f"**Intent:** `{intent}`\n\n**Channels:** `{channels}`")
            
            if "detected_fallacies" in raw_state and raw_state["detected_fallacies"]:
                st.markdown("**Interceptor Triggered**")
                st.warning(f"{len(raw_state['detected_fallacies'])} fallacies detected by mathematical guardrails.")
                
        with col2:
            st.markdown("**Deterministic Engine Outputs**")
            
            if raw_state.get("optimization_results"):
                st.success("✅ SLSQP Solver Executed")
                with st.expander("Raw Optimizer Payload"):
                    st.json(raw_state["optimization_results"])
                    
            if raw_state.get("audit_results"):
                st.success("✅ SRM / Power Audit Executed")
                with st.expander("Raw Audit Payload"):
                    st.json(raw_state["audit_results"])
                    
            if raw_state.get("calibration_results"):
                st.success("✅ PyMC MCMC Sampling Executed")
                with st.expander("Raw Calibration Payload"):
                    st.json(raw_state["calibration_results"])

        st.markdown("---")
        st.markdown("**Full AgentGraph State Payload**")
        
        # Filter out the full message history to prevent massive JSON rendering
        filtered_state = {k: v for k, v in raw_state.items() if k != "messages"}
        st.json(filtered_state)
