import numpy as np
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Optional

class HillSaturationModel:
    @staticmethod
    def revenue(spend: float | np.ndarray, v_max: float, k: float, n: float) -> float | np.ndarray:
        """
        R(S) = (V_max * S^n) / (K^n + S^n)
        """
        return (v_max * (spend ** n)) / ((k ** n) + (spend ** n))

    @staticmethod
    def marginal_roas(spend: float, v_max: float, k: float, n: float) -> float:
        """
        First derivative of the Hill saturation equation with respect to spend:
        dR/dS = (V_max * n * K^n * S^(n-1)) / (K^n + S^n)^2
        """
        # Safely handle near-zero spend where derivative might be zero or undefined
        if spend <= 1e-9:
            return 0.0
            
        numerator = v_max * n * (k ** n) * (spend ** (n - 1))
        denominator = ((k ** n) + (spend ** n)) ** 2
        return numerator / denominator

    @staticmethod
    def average_roas(spend: float, v_max: float, k: float, n: float) -> float:
        """
        aROAS = R(S) / S
        """
        if spend <= 1e-9:
            return 0.0
        return HillSaturationModel.revenue(spend, v_max, k, n) / spend


class BudgetOptimizer:
    def __init__(self, channel_params: Dict[str, Dict[str, float]]) -> None:
        """
        Initializes the optimizer with a dictionary of ground truth channel parameters.
        Format:
        {
            "tv": {"v_max": 800000, "k": 200000, "n": 1.5},
            ...
        }
        """
        self.channel_params = channel_params
        self.channels = list(channel_params.keys())

    def optimize_allocation(
        self, 
        total_budget: float, 
        bounds: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, Any]:
        """
        Uses SciPy SLSQP to find the optimal budget allocation across channels.
        Objective: Maximize total Hill revenue (minimizing negative total revenue).
        """
        n_channels = len(self.channels)
        
        # Initial guess: distribute budget equally
        initial_guess = np.full(n_channels, total_budget / n_channels)
        
        # Objective function (minimize negative total revenue)
        def objective(spends: np.ndarray) -> float:
            total_rev = 0.0
            for i, channel in enumerate(self.channels):
                p = self.channel_params[channel]
                total_rev += HillSaturationModel.revenue(spends[i], p["v_max"], p["k"], p["n"])
            return -total_rev
            
        # Equality Constraint: Sum of spends must exactly equal total_budget
        def budget_constraint(spends: np.ndarray) -> float:
            return np.sum(spends) - total_budget
            
        constraints = [{'type': 'eq', 'fun': budget_constraint}]
        
        # Bounds: [0, total_budget] by default unless user specified constraints
        scipy_bounds = []
        for ch in self.channels:
            if bounds and ch in bounds:
                scipy_bounds.append(bounds[ch])
            else:
                scipy_bounds.append((0, total_budget))
                
        # Run SLSQP Optimizer
        result = minimize(
            objective, 
            initial_guess, 
            method='SLSQP', 
            bounds=scipy_bounds, 
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        # Format the result payload
        allocations = {}
        expected_mroas = {}
        for i, ch in enumerate(self.channels):
            alloc_spend = float(result.x[i])
            p = self.channel_params[ch]
            allocations[ch] = alloc_spend
            expected_mroas[ch] = HillSaturationModel.marginal_roas(alloc_spend, p["v_max"], p["k"], p["n"])
            
        return {
            "allocations": allocations,
            "expected_revenue": float(-result.fun),
            "expected_mroas": expected_mroas,
            "status": result.message,
            "success": result.success
        }

    def calculate_reallocation_impact(
        self, 
        current_spends: Dict[str, float], 
        optimized_spends: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Computes budget delta, expected delta revenue, and total reallocation improvement.
        """
        current_rev = 0.0
        optimized_rev = 0.0
        channel_impact = {}
        
        for ch in self.channels:
            p = self.channel_params[ch]
            curr_s = current_spends.get(ch, 0.0)
            opt_s = optimized_spends.get(ch, 0.0)
            
            c_rev = HillSaturationModel.revenue(curr_s, p["v_max"], p["k"], p["n"])
            o_rev = HillSaturationModel.revenue(opt_s, p["v_max"], p["k"], p["n"])
            
            current_rev += c_rev
            optimized_rev += o_rev
            
            channel_impact[ch] = {
                "budget_delta": opt_s - curr_s,
                "revenue_delta": o_rev - c_rev
            }
            
        return {
            "total_improvement": optimized_rev - current_rev,
            "current_total_revenue": current_rev,
            "optimized_total_revenue": optimized_rev,
            "channel_impact": channel_impact
        }
