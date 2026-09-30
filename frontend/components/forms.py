import streamlit as st
from typing import Dict, Any, List

def render_budget_optimization_form() -> Dict[str, Any]:
    """Renders the budget optimizer configuration form."""
    with st.form(key="budget_optimization_form"):
        st.subheader("Budget Allocation Engine")
        st.markdown("Set your target budget and select which channels the optimizer should re-allocate.")
        
        total_budget = st.number_input(
            "Total Budget (€)", 
            min_value=10000.0, 
            max_value=10000000.0, 
            value=100000.0, 
            step=10000.0
        )
        
        target_channels = st.multiselect(
            "Channels to Optimize",
            options=["TV", "Google_Search", "Meta_Display", "TikTok", "YouTube"],
            default=["TV", "Google_Search", "Meta_Display", "TikTok", "YouTube"]
        )
        
        submit = st.form_submit_button("Run Allocation Engine")
        
        if submit:
            return {
                "submitted": True,
                "total_budget_eur": total_budget,
                "target_channels": target_channels
            }
        return {"submitted": False}

def render_experiment_audit_form() -> Dict[str, Any]:
    """Renders the A/B testing and Geo-experiment audit form."""
    with st.form(key="experiment_audit_form"):
        st.subheader("Experiment Integrity Auditor")
        st.markdown("Input raw traffic and conversions to test for Sample Ratio Mismatch (SRM).")
        
        col1, col2 = st.columns(2)
        with col1:
            control_traffic = st.number_input("Control Traffic", min_value=0, value=1000, step=100)
            control_conversions = st.number_input("Control Conversions", min_value=0, value=50, step=10)
        with col2:
            variant_traffic = st.number_input("Variant Traffic", min_value=0, value=1000, step=100)
            variant_conversions = st.number_input("Variant Conversions", min_value=0, value=60, step=10)
            
        alpha = st.selectbox("Significance Level (Alpha)", options=[0.01, 0.05, 0.10], index=1)
        
        submit = st.form_submit_button("Audit Experiment Validity")
        
        if submit:
            return {
                "submitted": True,
                "control_traffic": control_traffic,
                "variant_traffic": variant_traffic,
                "control_conversions": control_conversions,
                "variant_conversions": variant_conversions,
                "alpha": alpha
            }
        return {"submitted": False}

def render_calibration_form() -> Dict[str, Any]:
    """Renders the Bayesian calibration form to update MMM priors."""
    with st.form(key="calibration_form"):
        st.subheader("Bayesian Lift Calibration")
        st.markdown("Update your observational MMM prior using a causal experimental lift estimate.")
        
        channel = st.selectbox(
            "Target Channel", 
            options=["TV", "Google_Search", "Meta_Display", "TikTok", "YouTube"]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Observational Prior (MMM)**")
            prior_mean = st.number_input("Prior Mean (Lift)", value=1.5, step=0.1)
            prior_sd = st.number_input("Prior SD", value=0.5, step=0.05)
        with col2:
            st.markdown("**Experimental Estimate**")
            exp_mean = st.number_input("Experiment Lift Estimate", value=1.8, step=0.1)
            exp_se = st.number_input("Experiment Standard Error", value=0.2, step=0.05)
            
        submit = st.form_submit_button("Calibrate MMM Prior")
        
        if submit:
            return {
                "submitted": True,
                "channel": channel,
                "mmm_prior_mean": prior_mean,
                "mmm_prior_sd": prior_sd,
                "experiment_lift_mean": exp_mean,
                "experiment_lift_se": exp_se
            }
        return {"submitted": False}
