import pytest
from unittest.mock import patch, MagicMock
from src.api.dependencies import (
    get_data_engine,
    get_budget_optimizer,
    get_experiment_auditor,
    get_geo_auditor,
    get_bayesian_calibrator,
    get_mede_workflow
)
from src.tools.optimizer import BudgetOptimizer
from src.tools.experimentation import ExperimentAuditor, GeoExperimentAuditor
from src.tools.bayesian_calibrator import BayesianLiftCalibrator
from src.agents.graph import MEDEAgentWorkflow

@pytest.fixture(autouse=True)
def clear_dependency_caches():
    """Clear lru_cache before and after tests to prevent cross-test leakage."""
    get_data_engine.cache_clear()
    get_budget_optimizer.cache_clear()
    get_experiment_auditor.cache_clear()
    get_geo_auditor.cache_clear()
    get_bayesian_calibrator.cache_clear()
    get_mede_workflow.cache_clear()
    yield
    get_data_engine.cache_clear()
    get_budget_optimizer.cache_clear()
    get_experiment_auditor.cache_clear()
    get_geo_auditor.cache_clear()
    get_bayesian_calibrator.cache_clear()
    get_mede_workflow.cache_clear()

@patch("src.api.dependencies.DuckDBDataEngine")
@patch("src.core.config.get_settings")
def test_dependency_singletons(mock_get_settings, mock_duckdb):
    mock_settings = MagicMock()
    mock_settings.GOLD_DATA_PATH = "mock_path.parquet"
    mock_get_settings.return_value = mock_settings

    # Verify type correctness
    data_engine = get_data_engine()
    budget_optimizer = get_budget_optimizer()
    exp_auditor = get_experiment_auditor()
    geo_auditor = get_geo_auditor()
    bayesian_calibrator = get_bayesian_calibrator()

    assert data_engine == mock_duckdb.return_value
    assert isinstance(budget_optimizer, BudgetOptimizer)
    assert isinstance(exp_auditor, ExperimentAuditor)
    assert isinstance(geo_auditor, GeoExperimentAuditor)
    assert isinstance(bayesian_calibrator, BayesianLiftCalibrator)

    # Verify singleton / caching behavior (identity equality)
    assert get_data_engine() is data_engine
    assert get_budget_optimizer() is budget_optimizer
    assert get_experiment_auditor() is exp_auditor
    assert get_geo_auditor() is geo_auditor
    assert get_bayesian_calibrator() is bayesian_calibrator

@patch("src.api.dependencies.DuckDBDataEngine")
@patch("src.core.config.get_settings")
@patch("src.agents.nodes.synthesizer.ChatGoogleGenerativeAI")
@patch("src.agents.nodes.supervisor.ChatGoogleGenerativeAI")
def test_workflow_dependency_compilation(mock_supervisor_llm, mock_synth_llm, mock_get_settings, mock_duckdb):
    workflow_1 = get_mede_workflow()
    workflow_2 = get_mede_workflow()

    assert isinstance(workflow_1, MEDEAgentWorkflow)
    assert workflow_1.graph is not None
    # Verify compiled workflow is also cached as a singleton
    assert workflow_1 is workflow_2
