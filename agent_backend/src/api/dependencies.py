from functools import lru_cache
from src.core.config import get_settings
from src.tools.data_engine import DuckDBDataEngine
from src.tools.optimizer import BudgetOptimizer
from src.tools.experimentation import ExperimentAuditor, GeoExperimentAuditor
from src.tools.bayesian_calibrator import BayesianLiftCalibrator
from src.agents.graph import MEDEAgentWorkflow
from src.agents.nodes.supervisor import SupervisorNode
from src.agents.nodes.interceptor import FallacyInterceptorNode
from src.agents.nodes.auditor import CausalAuditorNode
from src.agents.nodes.synthesizer import SynthesizerNode

@lru_cache()
def get_data_engine() -> DuckDBDataEngine:
    settings = get_settings()
    return DuckDBDataEngine(parquet_path=settings.GOLD_DATA_PATH)

@lru_cache()
def get_budget_optimizer() -> BudgetOptimizer:
    # Default structural priors. In production, these would be fetched dynamically from the MMM.
    default_params = {
        "TV": {"v_max": 100000.0, "k": 50000.0, "n": 1.2},
        "Google_Search": {"v_max": 20000.0, "k": 10000.0, "n": 0.8},
        "Meta_Display": {"v_max": 40000.0, "k": 15000.0, "n": 1.5},
        "TikTok": {"v_max": 15000.0, "k": 8000.0, "n": 2.0},
        "YouTube": {"v_max": 30000.0, "k": 12000.0, "n": 1.1}
    }
    return BudgetOptimizer(channel_params=default_params)

@lru_cache()
def get_experiment_auditor() -> ExperimentAuditor:
    return ExperimentAuditor()

@lru_cache()
def get_geo_auditor() -> GeoExperimentAuditor:
    return GeoExperimentAuditor()

@lru_cache()
def get_bayesian_calibrator() -> BayesianLiftCalibrator:
    return BayesianLiftCalibrator(random_seed=42)

@lru_cache()
def get_mede_workflow() -> MEDEAgentWorkflow:
    # 1. Resolve deterministic engine dependencies
    data_engine = get_data_engine()
    budget_optimizer = get_budget_optimizer()
    experiment_auditor = get_experiment_auditor()
    geo_auditor = get_geo_auditor()
    bayesian_calibrator = get_bayesian_calibrator()

    # 2. Instantiate nodes with injected dependencies
    supervisor = SupervisorNode()
    
    interceptor = FallacyInterceptorNode(
        data_engine=data_engine, 
        budget_optimizer=budget_optimizer
    )
    
    auditor = CausalAuditorNode(
        experiment_auditor=experiment_auditor,
        bayesian_calibrator=bayesian_calibrator,
        geo_auditor=geo_auditor
    )
    
    synthesizer = SynthesizerNode()

    # 3. Wire the DAG
    return MEDEAgentWorkflow(
        supervisor=supervisor,
        interceptor=interceptor,
        auditor=auditor,
        synthesizer=synthesizer
    )
