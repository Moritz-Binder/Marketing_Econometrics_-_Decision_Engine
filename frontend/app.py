import streamlit as st
import os
from typing import Dict, Any

# Ensure we can import our components if running from root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.client import MEDEClient
from frontend.components.forms import (
    render_budget_optimization_form,
    render_experiment_audit_form,
    render_calibration_form
)
from frontend.components.charts import (
    render_hill_curves,
    render_allocation_comparison,
    render_calibration_distribution
)
from frontend.components.chat_panel import render_chat_history, render_executive_brief
from frontend.components.trace_viewer import render_trace_viewer

# -----------------------------------------------------------------------------
# App Initialization & State Management
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MEDE: Co-Pilot Workspace", 
    page_icon="📈", 
    layout="wide"
)

# Initialize the API client
if "client" not in st.session_state:
    st.session_state.client = MEDEClient()

# Initialize conversational state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize analytical state
if "current_brief" not in st.session_state:
    st.session_state.current_brief = None
if "current_raw_state" not in st.session_state:
    st.session_state.current_raw_state = None

def append_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

# -----------------------------------------------------------------------------
# Action Handlers
# -----------------------------------------------------------------------------
def handle_analyze_decision(query: str, total_budget: float = None, channels: list = None) -> bool:
    with st.spinner("Analyzing econometrics and routing intent..."):
        try:
            res = st.session_state.client.analyze_decision(query, total_budget, channels)
            # The API returns the ExecutiveDecisionBrief directly
            st.session_state.current_brief = res
            st.session_state.current_raw_state = res
            
            # Generate assistant response based on verdict
            verdict = st.session_state.current_brief.get("summary_verdict", "Analysis complete.")
            append_message("assistant", verdict)
            return True
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            append_message("assistant", f"API Error: {str(e)}")
            return False

def handle_audit(payload: Dict[str, Any]) -> bool:
    with st.spinner("Auditing statistical validity..."):
        try:
            res = st.session_state.client.audit_experiment(
                payload["control_traffic"], payload["variant_traffic"],
                payload["control_conversions"], payload["variant_conversions"],
                0.5, payload["alpha"]
            )
            # We mock a brief here since the direct endpoint doesn't return the full graph state
            verdict = res.get("verdict", "Audit Complete.")
            append_message("assistant", verdict)
            
            st.session_state.current_raw_state = {"audit_results": res}
            st.session_state.current_brief = {"summary_verdict": verdict}
            return True
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            append_message("assistant", f"API Error: {str(e)}")
            return False

def handle_calibrate(payload: Dict[str, Any]) -> bool:
    with st.spinner("Running MCMC Sampling (PyMC)..."):
        try:
            res = st.session_state.client.calibrate_channel(
                payload["channel"], payload["mmm_prior_mean"], payload["mmm_prior_sd"],
                payload["experiment_lift_mean"], payload["experiment_lift_se"]
            )
            verdict = res.get("executive_takeaway", "Calibration Complete.")
            append_message("assistant", verdict)
            
            st.session_state.current_raw_state = {"calibration_results": res}
            st.session_state.current_brief = {"summary_verdict": verdict}
            return True
        except Exception as e:
            st.error(f"API Error: {str(e)}")
            append_message("assistant", f"API Error: {str(e)}")
            return False

# -----------------------------------------------------------------------------
# UI Layout Layout: 2 Columns
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.title("🤖 MEDE Co-Pilot")
    st.markdown("Interact with the system naturally or use the exact control forms on the right.")
    
    # 1. Chat History
    chat_container = st.container(height=400)
    with chat_container:
        render_chat_history(st.session_state.messages)
    
    # Chat Input Trigger
    if prompt := st.chat_input("Ask MEDE to evaluate a budget shift or A/B test..."):
        append_message("user", prompt)
        success = handle_analyze_decision(prompt)
        if success:
            st.rerun()

    # 2. Executive Brief & Trace
    if st.session_state.current_brief:
        render_executive_brief(st.session_state.current_brief)
    
    if st.session_state.current_raw_state:
        render_trace_viewer(st.session_state.current_raw_state)

with col_right:
    st.title("📊 Analytical Workspace")
    
    tab_forms, tab_charts = st.tabs(["Interactive Parameters", "Analytical Diagnostics"])
    
    with tab_forms:
        # Form 1: Budget Optimization
        opt_payload = render_budget_optimization_form()
        if opt_payload["submitted"]:
            append_message("user", f"Run budget optimization for €{opt_payload['total_budget_eur']:,.2f} on {opt_payload['target_channels']}")
            success = handle_analyze_decision(
                query="Optimize my budget", 
                total_budget=opt_payload["total_budget_eur"], 
                channels=opt_payload["target_channels"]
            )
            if success:
                st.rerun()
            
        st.divider()
        
        # Form 2: Audit
        audit_payload = render_experiment_audit_form()
        if audit_payload["submitted"]:
            append_message("user", f"Audit A/B Test (Control: {audit_payload['control_conversions']}/{audit_payload['control_traffic']})")
            success = handle_audit(audit_payload)
            if success:
                st.rerun()
            
        st.divider()
        
        # Form 3: Calibration
        calib_payload = render_calibration_form()
        if calib_payload["submitted"]:
            append_message("user", f"Calibrate {calib_payload['channel']} MMM Prior with new experiment data.")
            success = handle_calibrate(calib_payload)
            if success:
                st.rerun()

    with tab_charts:
        raw_state = st.session_state.current_raw_state
        if raw_state:
            has_charts = False
            
            # Check for Optimization Results -> Render Hill Curves & Allocation Comparison
            if "optimization_results" in raw_state and raw_state["optimization_results"]:
                opt_res = raw_state["optimization_results"]
                if "channel_params" in opt_res and "allocations" in opt_res:
                    st.plotly_chart(
                        render_hill_curves(opt_res["channel_params"], opt_res.get("current_spends", {}), opt_res["allocations"]), 
                        use_container_width=True
                    )
                    st.plotly_chart(
                        render_allocation_comparison(opt_res.get("current_spends", {}), opt_res["allocations"]),
                        use_container_width=True
                    )
                    has_charts = True
                    
            # Check for Calibration Results -> Render Distribution
            if "calibration_results" in raw_state and raw_state["calibration_results"]:
                calib_res = raw_state["calibration_results"]
                st.plotly_chart(
                    render_calibration_distribution(
                        calib_res["original_prior_mean"], 0.5, # Assuming 0.5 as we don't return prior SD from API currently
                        calib_res["calibrated_posterior_mean"], calib_res["calibrated_posterior_sd"],
                        calib_res["hdi_95"]
                    ),
                    use_container_width=True
                )
                has_charts = True
                
            if not has_charts:
                st.info("No visualizations available for this execution trace. Run the Budget Optimizer or Calibration to see charts.")
        else:
            st.info("Awaiting execution data to render charts.")
