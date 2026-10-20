# MEDE Co-Pilot Workspace (Frontend)

[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0%2B-FF4B4B.svg?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.22.0%2B-3F4F75.svg?logo=plotly)](https://plotly.com)
[![httpx](https://img.shields.io/badge/httpx-0.27.0%2B-blue.svg)](https://www.python-httpx.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg?logo=docker)](Dockerfile)

The **MEDE Co-Pilot Workspace** is the executive decision interface for the Marketing Econometrics & Decision Engine. Designed specifically for Chief Marketing Officers (CMOs) and Brand Directors, it bridges the gap between natural language strategic inquiries and deterministic mathematical models.

---

## 💡 Core Architecture & Philosophy

### Zero Client-Side Calculation

In strict accordance with the MEDE architectural specification:

* **No Client Math:** The frontend never computes Hill saturation curves, adstock carryover, or posterior MCMC distributions.
* **Deterministic Visualization:** The UI strictly renders the exact coordinates, parameters, and distributions supplied by the FastAPI backend and deterministic analytical engines.
* **Non-Blocking Forms:** Parameter adjustments are encapsulated inside `st.form` blocks to prevent spurious re-renders on individual keystrokes or slider drags.

---

## 🖥️ Split-Screen Layout Architecture

The workspace implements a **two-column split-screen layout** (`layout="wide"`):

```text
┌────────────────────────────────────────┬────────────────────────────────────────┐
│  LEFT COLUMN: Conversational Co-Pilot   │  RIGHT COLUMN: Analytical Workspace    │
├────────────────────────────────────────┼────────────────────────────────────────┤
│  1. Chat History Container             │  Tabs:                                 │
│     - Multi-turn conversation display  │  ┌─────────────────┬─────────────────┐ │
│                                        │  │ 1. Parameters   │ 2. Diagnostics  │ │
│  2. Interactive Chat Input             │  ├─────────────────┴─────────────────┤ │
│     - Natural language queries         │  │ Interactive Forms:               │ │
│                                        │  │ • Budget Optimization Form        │ │
│  3. Executive Decision Brief           │  │ • A/B Experiment Audit Form       │ │
│     - Summary verdict                  │  │ • Bayesian Calibration Form       │ │
│     - Metric fallacy alert cards       │  ├──────────────────────────────────┤ │
│     - Recommended channel shifts       │  │ Plotly Diagnostic Charts:         │ │
│                                        │  │ • Non-linear Hill Saturation      │ │
│  4. Reasoning & Trace Expander         │  │ • Channel Budget Allocation Shifts│ │
│     - Collapsible DAG execution trace  │  │ • Bayesian Prior vs Posterior     │ │
│     - Solver convergence & status      │  │   with 95% HDI Shaded Bands       │ │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 📁 Component Directory Structure

```text
frontend/
├── README.md                  # Frontend Documentation (You are here)
├── Dockerfile                 # Multi-stage Streamlit container
├── requirements.txt           # Dependencies (Streamlit, Plotly, httpx, pydantic, scipy)
├── .env.example               # Frontend environment template
├── app.py                     # Main application entrypoint & session orchestrator
├── client.py                  # Typed HTTP client communicating with backend
├── components/
│   ├── __init__.py
│   ├── chat_panel.py          # Chat transcript and executive brief rendering
│   ├── forms.py               # Parameter input forms (Optimization, Audit, Calibration)
│   ├── charts.py              # Interactive Plotly chart builders
│   └── trace_viewer.py        # Collapsible execution trace and solver telemetry
└── tests/
    ├── __init__.py
    ├── test_client.py         # Unit tests for API client with mocked responses
    └── test_charts.py         # Unit tests validating Plotly figure generation
```

---

## ⚙️ Configuration & Environment Variables

Create your local `.env` file from the provided example:

```bash
cp .env.example .env
```

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `API_BASE_URL` | `http://localhost:8000/api/v1` | Root URL for the MEDE FastAPI backend service |
| `API_KEY_SECRET` | `mede-local-dev-key` | Shared API authentication key injected as `X-API-Key` |

---

## 🚀 Getting Started

### Local Development

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Backend Connection:**
   Ensure the backend service is running on `http://localhost:8000`. You can test this using the client health check:

   ```bash
   python -c "from frontend.client import MEDEClient; print('Backend Healthy:', MEDEClient().check_health())"
   ```

3. **Launch the Streamlit App:**

   ```bash
   streamlit run app.py --server.port 8501
   ```

   Open your browser at `http://localhost:8501`.

---

### Docker Execution

To run the frontend container independently:

```bash
docker build -t mede-frontend -f Dockerfile .
docker run -p 8501:8501 -e API_BASE_URL="http://host.docker.internal:8000/api/v1" mede-frontend
```

Alternatively, launch the full system via the root docker-compose file:

```bash
docker compose -f ../agent_backend/docker/docker-compose.yml up frontend_app -d
```

---

## 📊 Visual Diagnostic Modules

### 1. Non-Linear Hill Saturation Curves (`render_hill_curves`)

* Plots theoretical channel response curves ($R(S) = \frac{V_{\max} S^n}{K^n + S^n}$).
* Overlays scatter points representing **Current Spend** vs. **Optimal Spend**.
* Provides immediate visual intuition for over-saturated channels vs. under-invested growth drivers.

### 2. Budget Allocation & Shift Comparison (`render_allocation_comparison`)

* Side-by-side grouped bar chart displaying current spend vs recommended allocation per channel.
* Clear visual highlight of capital reallocation flow (e.g. trimming saturated TV spend and fueling high-marginal TikTok/Meta channels).

### 3. Bayesian MMM Prior Calibration (`render_calibration_distribution`)

* Plots Gaussian observational MMM prior density vs. updated MCMC posterior density.
* Highlights the **95% High Density Interval (HDI)** shaded band and shrinkage toward causal experimental truth.

---

## 🧪 Testing

Run frontend unit tests covering client endpoints and Plotly chart generation:

```bash
# Run tests from the frontend directory
PYTHONPATH=.. pytest tests/ -v
```
