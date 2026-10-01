import os
import httpx
from typing import Dict, Any, List, Optional

class MEDEClient:
    """
    Typed HTTP client for communicating with the MEDE FastAPI backend.
    """
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")
        self.timeout = 30.0
        self.api_key = os.environ.get("API_KEY_SECRET", "mede-local-dev-key")
        self.headers = {"X-API-Key": self.api_key}

    def check_health(self) -> bool:
        """Ping the health endpoint (which is at the root level)."""
        # The health route is usually at /health on the root app, so we strip /api/v1
        root_url = self.base_url.replace("/api/v1", "")
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{root_url}/health")
                return response.status_code == 200
        except httpx.RequestError:
            return False

    def analyze_decision(
        self, 
        query: str, 
        total_budget_eur: Optional[float] = None, 
        target_channels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Dispatch query to LangGraph execution endpoint."""
        payload = {"query": query}
        if total_budget_eur is not None:
            payload["total_budget_eur"] = total_budget_eur
        if target_channels is not None:
            payload["target_channels"] = target_channels

        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.post(f"{self.base_url}/analyze", json=payload)
            response.raise_for_status()
            return response.json()

    def audit_experiment(
        self, 
        control_traffic: int, 
        variant_traffic: int, 
        control_conversions: int, 
        variant_conversions: int, 
        planned_control_ratio: float = 0.5, 
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Dispatch statistical audit request."""
        payload = {
            "control_traffic": control_traffic,
            "variant_traffic": variant_traffic,
            "control_conversions": control_conversions,
            "variant_conversions": variant_conversions,
            "planned_control_ratio": planned_control_ratio,
            "alpha": alpha
        }
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.post(f"{self.base_url}/audit-experiment", json=payload)
            response.raise_for_status()
            return response.json()

    def calibrate_channel(
        self, 
        channel: str, 
        mmm_prior_mean: float, 
        mmm_prior_sd: float, 
        experiment_lift_mean: float, 
        experiment_lift_se: float
    ) -> Dict[str, Any]:
        """Dispatch Bayesian prior calibration request."""
        payload = {
            "channel": channel,
            "mmm_prior_mean": mmm_prior_mean,
            "mmm_prior_sd": mmm_prior_sd,
            "experiment_lift_mean": experiment_lift_mean,
            "experiment_lift_se": experiment_lift_se
        }
        with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
            response = client.post(f"{self.base_url}/calibrate", json=payload)
            response.raise_for_status()
            return response.json()
