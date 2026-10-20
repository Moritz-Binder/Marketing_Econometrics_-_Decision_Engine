# Marketing Econometrics & Decision Engine (MEDE) - Backend Service

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0%2B-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![DuckDB](https://img.shields.io/badge/DuckDB-1.0.0%2B-FFF000.svg?logo=duckdb)](https://duckdb.org)
[![PyMC](https://img.shields.io/badge/PyMC-5.15.0%2B-red.svg)](https://www.pymc.io)
[![SciPy](https://img.shields.io/badge/SciPy-1.13.0%2B-blue.svg)](https://scipy.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1.0%2B-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Prometheus](https://img.shields.io/badge/Prometheus-Enabled-E6522C.svg?logo=prometheus)](https://prometheus.io)

**MEDE Backend** is the analytical core and decision intelligence engine for the Marketing Econometrics & Decision Engine platform. It provides high-throughput, low-latency econometric reasoning, budget optimization, statistical auditing, and Bayesian MCMC calibration.

---

## 🛡️ Core Philosophy: Zero Hallucinated Arithmetic

In mission-critical enterprise marketing budget decisions, LLMs cannot be trusted with numerical computation. MEDE enforces strict architectural guardrails:

* **LLMs strictly route and synthesize:** LLMs classify commercial intent into typed routing schemas and translate mathematical outputs into executive decision briefs.
* **Math is 100% deterministic:** All econometric equations, carryover adstock transformations, Hill saturation curves, SLSQP budget allocations, Pearson Chi-Square tests, and PyMC Bayesian traces are executed strictly in deterministic Python and C/Fortran libraries.

---

## 🏛️ System Architecture

```mermaid
graph TD
    User([Client / Streamlit UI]) -->|HTTP POST + X-API-Key| Gateway[FastAPI Gateway :8000]
    Gateway --> Auth[verify_api_key Dependency]
    
    subgraph DirectEndpoints ["Direct Deterministic Endpoints"]
        Auth -->|POST /audit-experiment| ExpAuditor[Experiment Auditor]
        Auth -->|POST /calibrate| PyMC[Bayesian Calibrator]
    end
    
    subgraph CognitivePipeline ["LangGraph Execution Pipeline"]
        Auth -->|POST /analyze| Supervisor[Supervisor Node<br>Gemini 1.5 Flash<br>Structured Intent Classifier]
        
        Supervisor -->|BUDGET_OPTIMIZATION | FALLACY_CHECK| Interceptor[Fallacy Interceptor Node]
        Supervisor -->|AUDIT_EXPERIMENT | CALIBRATION| Auditor[Causal Auditor Node]
        
        Interceptor <--> DuckDB[DuckDB OLAP Engine<br>Out-of-Core Parquet / GCS httpfs]
        Interceptor <--> Optimizer[SciPy SLSQP Optimizer<br>Hill Non-Linear Curve]
        
        Auditor <--> ExpAuditor
        Auditor <--> PyMC
        Auditor <--> GeoAuditor[GeoExperiment Auditor<br>Synthetic Control Null Drift]
        
        Interceptor --> Synthesizer[Synthesizer Node<br>Gemini 1.5 Pro]
        Auditor --> Synthesizer
        Synthesizer --> Brief[ExecutiveDecisionBrief]
    end
    
    Brief --> Gateway
    Gateway --> User
```

---

## 🧮 Mathematical & Analytical Tool Modules (`src/tools/`)

### 1. `DuckDBDataEngine` & `AdstockTransformer` (`data_engine.py`)

* **Out-of-Core OLAP:** Scans multi-gigabyte historical marketing Parquet datasets without full in-memory loading.
* **Cloud Storage Ready:** Automatically loads the DuckDB `httpfs` extension to query Google Cloud Storage (`gs://...`) directly using Application Default Credentials (ADC).
* **Adstock Decay Transformations:**
  * **Geometric Decay:** $x_{\text{adstocked}}(t) = x(t) + \alpha \cdot x_{\text{adstocked}}(t-1)$
  * **Weibull PDF Adstock:** Normalizes Weibull probability density function increments and performs 1D convolution over spend.
  * **Weibull CDF Adstock:** Computes cumulative decay increments $w_t = F(t) - F(t-1)$ over lag windows.

### 2. `HillSaturationModel` & `BudgetOptimizer` (`optimizer.py`)

* **Hill Kinetics:**
  * Revenue: $R(S) = \frac{V_{\max} \cdot S^n}{K^n + S^n}$
  * Marginal ROAS: $\frac{dR}{dS} = \frac{V_{\max} \cdot n \cdot K^n \cdot S^{n-1}}{(K^n + S^n)^2}$
  * Average ROAS: $aROAS = \frac{R(S)}{S}$
* **Constrained Non-Linear Solver:** Solves the budget allocation problem via SciPy's Sequential Least Squares Programming (`SLSQP`) subject to $\sum S_i = \text{Budget}$ and bounds $L_i \le S_i \le U_i$.
* **Reallocation Impact:** Computes exact budget delta, marginal return deltas, and expected revenue improvement.

### 3. `ExperimentAuditor` & `GeoExperimentAuditor` (`experimentation.py`)

* **Sample Ratio Mismatch (SRM):** Pearson's Chi-Square Goodness-of-Fit test ($\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$) alerting stakeholders if traffic allocation is structurally corrupted ($p < \alpha$).
* **Statistical Power & MDE:** Computes power for two-sample proportion Z-tests and estimates minimum detectable effect sizes and required sample sizes per arm.
* **GeoExperiment Null Drift Simulation:** Runs bootstrap permutations on historical geo-sales control/treatment series to establish empirical false-positive distributions and pre-test MDE.

### 4. `BayesianLiftCalibrator` & `LiftCorrectionEngine` (`bayesian_calibrator.py`)

* **PyMC MCMC Sampling:** Combines observational MMM normal priors with causal experiment lift measurements:

  ```python
  with pm.Model() as model:
      beta = pm.Normal("beta_channel", mu=prior_mean, sigma=prior_sd)
      obs = pm.Normal("observed_lift", mu=beta, sigma=lift_se, observed=lift_estimate)
      trace = pm.sample(draws=2000, tune=1000, return_inferencedata=True)
  ```

* **ArviZ Integration:** Extracts posterior mean, posterior standard deviation, 95% High Density Interval (HDI), and empirical shrinkage metrics.
* **Lift Corrections:** Corrects raw experimental lift for adstock tail carryover effects and non-linear test-cell saturation.

---

## 🤖 LangGraph Multi-Agent Architecture (`src/agents/`)

The agent graph maintains typed execution state (`AgentGraphState`) across four specialized nodes:

1. **`SupervisorNode`:** Uses `gemini-1.5-flash` with Pydantic structured output (`SupervisorIntentRouting`) to categorize stakeholder queries into `BUDGET_OPTIMIZATION`, `AUDIT_EXPERIMENT`, `FALLACY_CHECK`, or `CALIBRATION`, while extracting target marketing channels and parameters.
2. **`FallacyInterceptorNode`:** Interrogates DuckDB data and the SLSQP optimizer to detect:
   * *Average vs. Marginal Confusion* ($aROAS \gg 1$ but $mROAS < 1$).
   * *Adstock/Decay Blindness* (pausing spend without observing immediate drop).
   * *Over-saturation Thresholds* ($S > K_{0.5}$).
3. **`CausalAuditorNode`:** Dispatches deterministic statistical testing (SRM Chi-Square, Z-test power, or PyMC Bayesian updating) based on extracted experiment parameters.
4. **`SynthesizerNode`:** Uses `gemini-1.5-pro` with `.with_structured_output(ExecutiveDecisionBrief)` to assemble executive verdicts, risk ratings, metric corrections, and methodological caveats without altering deterministic arithmetic.

---

## ⚙️ Configuration & Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PROJECT_NAME` | `"Marketing Econometrics & Decision Engine (MEDE)"` | Service title |
| `VERSION` | `"1.0.0"` | API version |
| `API_V1_STR` | `"/api/v1"` | API prefix |
| `GEMINI_API_KEY` | *(Required)* | Google Gemini API key for LangGraph reasoning |
| `ROUTER_MODEL` | `"gemini-1.5-flash"` | Low-latency LLM model for supervisor intent classification |
| `REASONING_MODEL` | `"gemini-1.5-pro"` | High-capability LLM model for executive brief synthesis |
| `API_KEY_SECRET` | `"mede-local-dev-key"` | Secret key required in the `X-API-Key` request header |
| `GOLD_DATA_PATH` | `"data/gold/marketing_mix_gold.parquet"` | Path to the gold Parquet dataset (local path or `gs://...`) |
| `LANGFUSE_PUBLIC_KEY` | `None` | (Optional) Langfuse observability public key |
| `LANGFUSE_SECRET_KEY` | `None` | (Optional) Langfuse observability secret key |
| `LANGFUSE_HOST` | `"https://cloud.langfuse.com"` | Langfuse host URL |
| `PROMETHEUS_PORT` | `9090` | Internal Prometheus metric exporter port |

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites

* Python 3.11+
* Active `GEMINI_API_KEY`

### 2. Local Execution

```bash
# 1. Install dependencies in editable mode
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# 2. Generate Synthetic 156-Week FMCG Parquet Database
python scripts/generate_synthetic_data.py

# 3. Configure environment
cp .env.example .env
# Edit .env and supply GEMINI_API_KEY

# 4. Launch FastAPI development server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Docker Compose Execution

To launch backend, frontend, and Prometheus observability stack:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

---

## 📡 API Endpoints & Usage

All `/api/v1/*` routes require the `X-API-Key` authentication header.

### 1. Analyze Decision & Fallacy Check (`POST /api/v1/analyze`)

Dispatches the query through the LangGraph cognitive engine and deterministic tools:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mede-local-dev-key" \
  -d '{
    "query": "TV has an average ROAS of 3.5. Let us double our TV investment.",
    "total_budget_eur": 1000000,
    "target_channels": ["TV"]
  }'
```

**Response (`ExecutiveDecisionBrief`):**

```json
{
  "summary_verdict": "CRITICAL RISK: Do not double the TV budget based on historical average ROAS.",
  "fallacies": [
    {
      "metric": "ROAS",
      "user_assumption": "High Average ROAS implies High Marginal ROAS",
      "mathematical_reality": "TV is operating deep in the saturation plateau. Marginal ROAS is below 1.0.",
      "risk_level": "Critical",
      "capital_at_risk_eur": 500000.0
    }
  ],
  "recommended_allocations": [
    {
      "channel": "TV",
      "current_spend_eur": 400000.0,
      "recommended_spend_eur": 210000.0,
      "expected_mroas": 1.12,
      "historical_aroas": 3.5,
      "saturation_state": "Over-saturated"
    }
  ],
  "methodology_caveats": [
    "Assumes steady-state competitor promotional intensity.",
    "Decay half-life estimated via geometric carryover (alpha=0.7)."
  ]
}
```

---

### 2. Audit Experiment Validity (`POST /api/v1/audit-experiment`)

Performs deterministic Sample Ratio Mismatch (SRM) and statistical power audits:

```bash
curl -X POST http://localhost:8000/api/v1/audit-experiment \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mede-local-dev-key" \
  -d '{
    "control_traffic": 50000,
    "variant_traffic": 48200,
    "control_conversions": 2500,
    "variant_conversions": 2650,
    "planned_control_ratio": 0.5,
    "alpha": 0.05
  }'
```

**Response (`ExperimentAuditResponse`):**

```json
{
  "srm_detected": true,
  "srm_p_value": 0.00001,
  "chi2_stat": 33.02,
  "is_valid_test": false,
  "statistical_power": 0.94,
  "verdict": "INVALID TEST: Severe Sample Ratio Mismatch detected (p=0.0000). Do not use these results."
}
```

---

### 3. Calibrate Channel Prior (`POST /api/v1/calibrate`)

Executes PyMC MCMC updating between MMM observational priors and causal lift experiments:

```bash
curl -X POST http://localhost:8000/api/v1/calibrate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mede-local-dev-key" \
  -d '{
    "channel": "TV",
    "mmm_prior_mean": 2.20,
    "mmm_prior_sd": 0.40,
    "experiment_lift_mean": 1.35,
    "experiment_lift_se": 0.15
  }'
```

**Response (`CalibrationResponse`):**

```json
{
  "channel": "TV",
  "original_prior_mean": 2.2,
  "calibrated_posterior_mean": 1.458,
  "calibrated_posterior_sd": 0.139,
  "hdi_95": [1.185, 1.732],
  "shrinkage_factor": 0.879,
  "executive_takeaway": "MMM prior for TV updated from 2.2 to 1.4580."
}
```

---

### 4. Health & Metrics Endpoints

* **Health Check:** `curl http://localhost:8000/health`

  ```json
  {"status": "healthy", "version": "1.0.0"}
  ```

* **Prometheus Metrics:** `curl http://localhost:8000/metrics`

---

## 🧪 Testing

Execute the comprehensive test suite (42 unit and integration tests):

```bash
# Run pytest with source root in PYTHONPATH
PYTHONPATH=. .venv/bin/pytest tests -v
```

Test coverage includes:

* **Unit Tests:** `test_adstock.py`, `test_optimizer.py`, `test_experimentation.py`, `test_bayesian_calibrator.py`, `test_synthetic_data.py`, `test_config.py`, `test_telemetry.py`, `test_dependencies.py`, `test_schemas.py`, `test_supervisor.py`, `test_interceptor.py`, `test_auditor.py`, `test_synthesizer.py`.
* **Integration Tests:** `test_api.py` (FastAPI contracts), `test_graph_flow.py` (LangGraph state flow), `test_main.py` (lifespan and pre-warming).

---

## 🔬 Benchmark & Evaluation Suite

Run the 20-case commercial benchmark evaluation pipeline:

```bash
python evaluations/run_evals.py
```

Measures:

* Fallacy Detection Accuracy ($\ge 95\%$)
* SRM Recall ($100\%$)
* Zero-Hallucination Compliance (verifying that reported metrics match deterministic engine values)
* Pydantic Schema Adherence ($100\%$)
