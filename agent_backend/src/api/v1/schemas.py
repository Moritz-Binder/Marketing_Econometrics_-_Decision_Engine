from typing import List, Optional, Literal
from pydantic import BaseModel, Field

# --- Requests ---

class AnalyzeDecisionRequest(BaseModel):
    query: str = Field(..., description="The executive's commercial query.")
    total_budget_eur: Optional[float] = Field(None, description="Optional hard constraint for optimization.")
    target_channels: Optional[List[str]] = Field(None, description="Channels in scope.")

class ExperimentAuditRequest(BaseModel):
    control_traffic: int
    variant_traffic: int
    control_conversions: int
    variant_conversions: int
    planned_control_ratio: float = 0.5
    alpha: float = 0.05

class CalibrationRequest(BaseModel):
    channel: str
    mmm_prior_mean: float
    mmm_prior_sd: float
    experiment_lift_mean: float
    experiment_lift_se: float

# --- Responses ---

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

class ExperimentAuditResponse(BaseModel):
    srm_detected: bool
    srm_p_value: float
    chi2_stat: float
    is_valid_test: bool
    statistical_power: float
    verdict: str

class CalibrationResponse(BaseModel):
    channel: str
    original_prior_mean: float
    calibrated_posterior_mean: float
    calibrated_posterior_sd: float
    hdi_95: List[float]
    shrinkage_factor: float
    executive_takeaway: str

class ExecutiveDecisionBrief(BaseModel):
    summary_verdict: str
    fallacies: List[MetricCorrection]
    recommended_allocations: List[ReallocationItem]
    methodology_caveats: List[str]
