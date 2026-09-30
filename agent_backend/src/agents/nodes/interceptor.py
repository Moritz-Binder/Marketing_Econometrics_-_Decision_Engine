from typing import Dict, Any
from src.agents.state import AgentGraphState
from src.tools.data_engine import DuckDBDataEngine
from src.tools.optimizer import BudgetOptimizer, HillSaturationModel
import pandas as pd

class FallacyInterceptorNode:
    def __init__(self, data_engine: DuckDBDataEngine, budget_optimizer: BudgetOptimizer) -> None:
        self.data_engine = data_engine
        self.budget_optimizer = budget_optimizer
        
    def __call__(self, state: AgentGraphState) -> Dict[str, Any]:
        intent = state.get("user_intent", "")
        channels = state.get("target_channels", [])
        
        # We only run fallacy checks for certain intents
        if intent not in ["BUDGET_OPTIMIZATION", "FALLACY_CHECK"]:
            return {}
            
        detected_fallacies = state.get("detected_fallacies", []) or []
        
        # Get historical summaries from DuckDB
        summary_df = self.data_engine.get_channel_summary()
        
        for channel in channels:
            if channel not in self.budget_optimizer.channel_params:
                continue
                
            params = self.budget_optimizer.channel_params[channel]
            v_max = params["v_max"]
            k = params["k"]
            n = params["n"]
            alpha = params.get("alpha", 0.0)
            
            # Fetch historical spend for the channel to evaluate its current saturation state
            channel_data = summary_df[summary_df["channel"] == channel]
            if channel_data.empty:
                continue
                
            # For this evaluation, we use the average weekly spend.
            # Assuming our synthetic dataset is 156 weeks.
            weekly_spend = channel_data.iloc[0]["total_spend"] / 156.0
            historical_aroas = channel_data.iloc[0]["aroas"]
            
            # Compute current marginal ROAS at the current weekly spend level
            mroas = HillSaturationModel.marginal_roas(weekly_spend, v_max, k, n)
            
            # 1. Average vs. Marginal Confusion
            # If the user sees high historical aROAS but current marginal ROAS is saturated (< 1.0)
            if historical_aroas > 1.5 and mroas < 1.0:
                detected_fallacies.append({
                    "metric": "Average vs. Marginal ROAS",
                    "user_assumption": f"Scale {channel} because historical ROAS is high ({historical_aroas:.2f}).",
                    "mathematical_reality": f"The channel is saturated. The next dollar spent yields only €{mroas:.2f}.",
                    "risk_level": "High",
                    "capital_at_risk_eur": float(weekly_spend * (1 - mroas))
                })
                
            # 2. Over-saturation Threshold
            # If they are spending past the inflection point K
            if weekly_spend > k:
                detected_fallacies.append({
                    "metric": "Over-saturation Threshold",
                    "user_assumption": f"{channel} spend will continue to scale efficiently.",
                    "mathematical_reality": f"Weekly spend (€{weekly_spend:,.0f}) exceeds the saturation inflection point K (€{k:,.0f}).",
                    "risk_level": "Medium",
                    "capital_at_risk_eur": float(weekly_spend - k)
                })
                
            # 3. Adstock / Decay Blindness
            # If the user intent is FALLACY_CHECK and they mention cutting a channel.
            if alpha > 0.6 and intent == "FALLACY_CHECK":
                detected_fallacies.append({
                    "metric": "Adstock Decay Blindness",
                    "user_assumption": f"Cutting {channel} budget is safe because sales don't drop immediately.",
                    "mathematical_reality": f"{channel} has high momentum (decay={alpha:.2f}). Sales will collapse slowly over months, masking the true loss.",
                    "risk_level": "Critical",
                    "capital_at_risk_eur": float(v_max * 0.3)
                })
                
        optimization_results = None
        if intent == "BUDGET_OPTIMIZATION":
            extracted = state.get("extracted_parameters", {})
            total_budget = extracted.get("total_budget_eur") if extracted else None
            
            if total_budget:
                opt_res = self.budget_optimizer.optimize_allocation(total_budget=total_budget)
                
                allocs = []
                for ch, rec_spend in opt_res["allocations"].items():
                    # Check if this channel was targeted (if specific targets were given)
                    if channels and ch not in channels:
                        continue
                        
                    ch_data = summary_df[summary_df["channel"] == ch]
                    curr_spend = float(ch_data.iloc[0]["total_spend"] / 156.0) if not ch_data.empty else 0.0
                    hist_aroas = float(ch_data.iloc[0]["aroas"]) if not ch_data.empty else 0.0
                    
                    params = self.budget_optimizer.channel_params.get(ch, {})
                    if params:
                        mroas = HillSaturationModel.marginal_roas(rec_spend, params["v_max"], params["k"], params["n"])
                        if rec_spend < params["k"] * 0.8:
                            sat = "Under-saturated"
                        elif rec_spend > params["k"] * 1.2:
                            sat = "Over-saturated"
                        else:
                            sat = "Optimal"
                    else:
                        mroas = 0.0
                        sat = "Optimal"
                        
                    allocs.append({
                        "channel": ch,
                        "current_spend_eur": curr_spend,
                        "recommended_spend_eur": float(rec_spend),
                        "expected_mroas": float(mroas),
                        "historical_aroas": hist_aroas,
                        "saturation_state": sat
                    })
                    
                optimization_results = {
                    "status": opt_res["status"],
                    "allocations": allocs,
                    "expected_revenue": float(opt_res["expected_revenue"])
                }
                
        return {
            "detected_fallacies": detected_fallacies,
            "optimization_results": optimization_results
        }
