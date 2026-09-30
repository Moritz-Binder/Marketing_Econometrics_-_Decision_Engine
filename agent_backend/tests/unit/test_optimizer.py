import numpy as np
from src.tools.optimizer import HillSaturationModel, BudgetOptimizer

def test_hill_functions():
    # V_max = 100, K = 50, n = 2
    # At S = 50: 
    # R(S) = 100 * (50^2) / (50^2 + 50^2) = 100 * 2500 / 5000 = 50
    rev = HillSaturationModel.revenue(50, v_max=100, k=50, n=2)
    assert np.isclose(rev, 50.0)
    
    # aROAS = 50 / 50 = 1.0
    aroas = HillSaturationModel.average_roas(50, v_max=100, k=50, n=2)
    assert np.isclose(aroas, 1.0)
    
    # dR/dS at S=50
    # (100 * 2 * 2500 * 50) / (5000^2) = 25,000,000 / 25,000,000 = 1.0
    mroas = HillSaturationModel.marginal_roas(50, v_max=100, k=50, n=2)
    assert np.isclose(mroas, 1.0)

def test_budget_optimizer():
    params = {
        "channel_a": {"v_max": 200, "k": 100, "n": 2}, # Steeper curve initially
        "channel_b": {"v_max": 50, "k": 10, "n": 1.5}  # Flatter curve, saturates quickly
    }
    
    optimizer = BudgetOptimizer(params)
    
    # Test allocation
    result = optimizer.optimize_allocation(total_budget=100)
    assert result["success"] is True
    
    alloc_a = result["allocations"]["channel_a"]
    alloc_b = result["allocations"]["channel_b"]
    
    # Budget constraint holds
    assert np.isclose(alloc_a + alloc_b, 100.0)
    
    # Check marginal ROAS equilibrium (they should be perfectly balanced at optimality)
    mroas_a = result["expected_mroas"]["channel_a"]
    mroas_b = result["expected_mroas"]["channel_b"]
    assert np.isclose(mroas_a, mroas_b, atol=0.01)

def test_reallocation_impact():
    params = {
        "channel_a": {"v_max": 200, "k": 100, "n": 2}
    }
    optimizer = BudgetOptimizer(params)
    
    current = {"channel_a": 50}    # R = 40
    optimized = {"channel_a": 100} # R = 200 * 10000 / 20000 = 100
    
    impact = optimizer.calculate_reallocation_impact(current, optimized)
    assert np.isclose(impact["total_improvement"], 60.0) # 100 - 40