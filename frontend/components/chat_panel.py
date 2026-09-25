import streamlit as st
from typing import List, Dict, Any

def render_chat_history(messages: List[Dict[str, str]]):
    """
    Renders the chat history.
    messages is a list of dictionaries with 'role' ('user' or 'assistant') and 'content'.
    """
    if not messages:
        st.info("No conversation history. Ask MEDE a question or submit a form to begin.")
        return

    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_executive_brief(brief: Dict[str, Any]):
    """
    Renders the structured ExecutiveDecisionBrief payload.
    """
    if not brief:
        return
        
    st.markdown("---")
    st.header("Executive Summary")
    
    # Summary Verdict
    if "summary_verdict" in brief:
        st.info(f"**Verdict:** {brief['summary_verdict']}")
        
    # Reallocation Recommendations
    allocations = brief.get("recommended_allocations", [])
    if allocations:
        st.subheader("Budget Recommendations")
        for alloc in allocations:
            channel = alloc.get("channel", "Unknown")
            curr = alloc.get("current_spend_eur", 0)
            rec = alloc.get("recommended_spend_eur", 0)
            saturation = alloc.get("saturation_state", "")
            mroas = alloc.get("expected_mroas", 0)
            
            delta = rec - curr
            direction = "🔺" if delta > 0 else "🔻" if delta < 0 else "➖"
            
            st.markdown(f"""
            **{channel}** {direction} (Shift: €{delta:,.2f})
            - New Target: €{rec:,.2f}
            - Expected mROAS: {mroas:.2f}
            - State: {saturation}
            """)
            
    # Fallacies / Corrections
    fallacies = brief.get("fallacies", [])
    if fallacies:
        st.subheader("⚠️ Methodological Corrections")
        for f in fallacies:
            risk = f.get("risk_level", "Medium")
            color = "red" if risk in ["High", "Critical"] else "orange"
            
            with st.expander(f"[{risk} Risk] Fallacy: {f.get('metric', 'Metric')}"):
                st.markdown(f"**User Assumption:** {f.get('user_assumption', '')}")
                st.markdown(f":{color}[**Econometric Reality:**] {f.get('mathematical_reality', '')}")
                if "capital_at_risk_eur" in f:
                    st.markdown(f"**Capital at Risk:** €{f['capital_at_risk_eur']:,.2f}")
                    
    # Caveats
    caveats = brief.get("methodology_caveats", [])
    if caveats:
        st.subheader("Methodology Caveats")
        for c in caveats:
            st.markdown(f"- {c}")
