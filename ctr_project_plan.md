# CTR Prediction Project — Industry-Style Build Plan

> **Goal:** Build an end-to-end ML pipeline that predicts click-through rate (CTR) on a content recommendation surface, with the engineering rigor of an industry production system.
> **Outcome:** Resume-defensible project where every interview question from the previous doc has a concrete, demonstrable answer.

---

## How to Read This Document

The plan is split into **six phases**, each ending in a concrete deliverable. You can:

- Stop after **Phase 3** and have a solid resume project (~2 weeks of work)
- Continue through **Phase 6** for a production-grade portfolio piece (~4-6 weeks)
- Skip cloud phases entirely if running locally only — every phase has a "local-only" path

**Phases at a glance:**

| Phase | Name | Time | What you'll have |
|---|---|---|---|
| 0 | Setup & Project Scaffolding | 1-2 days | Clean repo structure, env, data |
| 1 | EDA & Feature Engineering | 3-4 days | Notebooks, feature definitions |
| 2 | Modeling & Offline Evaluation | 3-4 days | Trained model, MLflow runs |
| 3 | Pipeline Orchestration | 3-5 days | Airflow/Prefect DAG running E2E |
| 4 | Serving & API | 2-3 days | FastAPI service with model |
| 5 | Monitoring & Drift Detection | 2-3 days | Dashboards, drift alerts |
| 6 | A/B Test Simulation & Documentation | 2-3 days | Mock A/B test, README, demo |

---

## Industry-Standard Tool Stack

This is the stack I'd use if I were starting a CTR project at a mid-sized tech company today. Each tool has a "why we use it" reason and a "lighter alternative if you're learning."

### Core stack

| Layer | Industry tool | Lighter alternative | Why |
|---|---|---|---|
| **Language** | Python 3.10+ | — | Universal in ML |
| **Env mgmt** | `uv` or `poetry` | `pip + venv` | Reproducible deps |
| **Data processing** | PySpark (large) / Polars | Pandas | Scale matters in interviews |
| **Modeling** | LightGBM, XGBoost | scikit-learn | Industry standard for tabular |
| **Experiment tracking** | MLflow | Weights & Biases | Open-source, easy to self-host |
| **Feature store** | Feast | SQLite + Parquet | Real feature stores cost $$ |
| **Orchestration** | Airflow | Prefect / Dagster | Airflow is the most common in industry |
| **Serving** | FastAPI + Docker | Flask | FastAPI is the modern default |
| **Monitoring** | Evidently AI + Grafana | Custom Streamlit dashboard | Evidently is purpose-built for ML drift |
| **Storage** | S3 / MinIO | Local filesystem | MinIO = local S3 |
| **Database** | PostgreSQL | SQLite | Postgres if you'll showcase SQL |
| **Containerization** | Docker + docker-compose | — | Non-negotiable for portfolio |
| **CI/CD** | GitHub Actions | — | Free for public repos |
| **Cloud (optional)** | AWS / GCP free tier | — | Skip if local-only |

### Why this stack signals "I've done this before"

In interviews, it's not about naming the right tools — it's about explaining why you'd choose one over another. With this stack, you can do exactly that.:

- *"We used MLflow for experiment tracking; we considered W&B but chose MLflow because we wanted self-hosted artifact storage."*
- *"Airflow for orchestration — it's overkill for a small DAG but the team wanted SLA monitoring and the ability to backfill arbitrary date ranges."*
- *"LightGBM over XGBoost — comparable accuracy, faster training on our wide feature set."*

That's exactly what experienced interviewers want to hear.

---

## Repository Structure (industry-standard)

Set this up on day one. A clean repo structure is a green flag in code reviews.

```
ctr-prediction/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # lint, test on every PR
│       └── train.yml               # weekly retraining trigger
├── config/
│   ├── data.yaml                   # data paths, schemas
│   ├── features.yaml               # feature definitions
│   ├── model.yaml                  # model hyperparameters
│   └── serving.yaml                # serving config
├── data/                           # gitignored
│   ├── raw/                        # original generated data
│   ├── interim/                    # cleaned, joined
│   ├── processed/                  # feature-engineered, ready for training
│   └── features/                   # feature store offline files
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_exploration.ipynb
│   ├── 03_model_baseline.ipynb
│   └── 04_error_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── ingest.py              # load raw data
│   │   ├── validate.py            # schema validation (Great Expectations / Pandera)
│   │   └── splits.py              # time-based train/val/test
│   ├── features/
│   │   ├── user_features.py
│   │   ├── item_features.py
│   │   ├── interaction_features.py
│   │   └── pipeline.py            # orchestrates feature build
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── calibrate.py
│   │   └── registry.py            # MLflow model registry interactions
│   ├── serving/
│   │   ├── app.py                 # FastAPI app
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── predictor.py           # model loading + inference
│   ├── monitoring/
│   │   ├── drift.py               # Evidently drift checks
│   │   └── performance.py         # daily metric computation
│   └── utils/
│       ├── logging.py
│       └── io.py
├── pipelines/
│   ├── airflow/
│   │   └── dags/
│   │       ├── feature_etl_dag.py
│   │       ├── training_dag.py
│   │       └── monitoring_dag.py
│   └── docker-compose.yml         # local Airflow + Postgres + MLflow
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_serving.py
├── scripts/
│   ├── generate_data.py           # the data generator I built for you
│   └── ab_test_simulation.py
├── docker/
│   ├── Dockerfile.training
│   ├── Dockerfile.serving
│   └── Dockerfile.airflow
├── docs/
│   ├── architecture.md            # system diagram + write-up
│   ├── model_card.md              # the trained model's "spec sheet"
│   └── ab_test_results.md
├── .gitignore
├── pyproject.toml                 # deps + tool config
├── Makefile                       # `make train`, `make serve`, etc.
└── README.md
```

> **Pro tip:** Push the empty scaffold to GitHub on day 1. Every commit after that becomes part of your project narrative. Interviewers sometimes look at commit history.

---

## Phase 0: Setup & Scaffolding (1-2 days)

### Tasks

1. **Initialize git repo + push to GitHub** (public)
2. **Set up Python environment** with `uv` or `poetry`
   ```bash
   uv init ctr-prediction
   uv add pandas numpy lightgbm scikit-learn fastapi uvicorn mlflow \
          evidently great-expectations pyyaml polars feast
   uv add --dev pytest ruff black mypy ipykernel jupyter
   ```
3. **Create the directory structure** above (empty files are fine)
4. **Run the data generator** I gave you — produces `data/raw/*.csv`
5. **Set up pre-commit hooks** for linting (black + ruff)
6. **Write a Makefile** with shortcuts:
   ```makefile
   data:
       python scripts/generate_data.py
   features:
       python -m src.features.pipeline
   train:
       python -m src.models.train
   serve:
       uvicorn src.serving.app:app --reload
   test:
       pytest tests/
   ```
7. **First README commit** — even a stub. Mentions: the problem, the approach, how to run.

### Deliverable

A repo that anyone can clone, run `make data && make train`, and have a working baseline.

---

## Phase 1: EDA & Feature Engineering (3-4 days)

### Tasks

#### Day 1: EDA notebook (`notebooks/01_eda.ipynb`)

- Load impressions, clicks, users, items
- Compute and visualize:
  - Overall CTR
  - CTR by category, device, country, hour of day, position
  - Distribution of impressions per user (long tail check)
  - Item age distribution
  - Missing data audit
- **Document findings** — this becomes ammo for the "what was the data like" interview question

#### Day 2: Data validation layer

Use **Pandera** or **Great Expectations** to define schemas:

```python
# src/data/validate.py
import pandera as pa
from pandera import Column, Check

impression_schema = pa.DataFrameSchema({
    "impression_id": Column(str, unique=True),
    "user_id": Column(str),
    "item_id": Column(str),
    "timestamp": Column(pa.DateTime),
    "position": Column(int, Check.ge(0)),
    "clicked": Column(int, Check.isin([0, 1])),
})
```

This is what real teams do. It catches "the upstream pipeline silently changed the schema" disasters.

#### Day 3-4: Feature engineering

Build features in **three categories** (matches the interview answers):

**User features** (`src/features/user_features.py`):
- `user_ctr_7d`, `user_ctr_30d` — historical click rates
- `user_session_count_7d`
- `user_recency_hours` — hours since last impression
- `user_top_category` — most-engaged category
- `user_age_bucket`, `user_country` (passthrough)

**Item features** (`src/features/item_features.py`):
- `item_ctr_7d`, `item_ctr_lifetime` (with smoothing for new items)
- `item_age_hours` (at impression time — point-in-time correctness!)
- `item_category`, `item_source` (one-hot or target-encoded)
- `item_has_image`, `item_has_video`

**Interaction features** (`src/features/interaction_features.py`):
- `user_x_category_ctr_30d` — user's CTR within this item's category
- `user_x_source_impressions_7d` — has user seen this source before
- `item_position_avg` — average position this item appears at

**Critical:** Every feature must be computed using **only data available before the impression timestamp**. This is point-in-time correctness — the most common bug in CTR projects.

```python
# Pattern: feature value at impression time = aggregation over data BEFORE that time
def user_ctr_7d(impressions_df, target_timestamp):
    window_start = target_timestamp - pd.Timedelta(days=7)
    past = impressions_df[
        (impressions_df["timestamp"] < target_timestamp) &
        (impressions_df["timestamp"] >= window_start)
    ]
    return past.groupby("user_id")["clicked"].mean()
```

### Deliverable

- EDA notebook with insights documented
- Feature pipeline that produces `data/processed/training_data.parquet`
- Schema validation passing on all stages

---

## Phase 2: Modeling & Offline Evaluation (3-4 days)

### Tasks

#### Day 1: Train/val/test splits — TIME-BASED

```python
# src/data/splits.py
def time_based_split(df, train_end, val_end):
    """
    Train: data BEFORE train_end
    Val:   train_end to val_end
    Test:  AFTER val_end
    """
    train = df[df["timestamp"] < train_end]
    val   = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)]
    test  = df[df["timestamp"] >= val_end]
    return train, val, test
```

For the 30-day generated data: train on days 1-21, validate on days 22-25, test on days 26-30.

**Never random splits.** This is the #1 leakage source in CTR projects. Make sure your README mentions this explicitly — it's a senior-level signal.

#### Day 2: Baseline + LightGBM

Start with a baseline so you can show lift:

1. **Naive baseline** — predict global CTR for everyone (~3%). Shows AUC = 0.5.
2. **Logistic regression** — sanity check + linear baseline.
3. **LightGBM** — your main model.

```python
# src/models/train.py
import lightgbm as lgb
import mlflow

with mlflow.start_run(run_name="lgbm_baseline"):
    mlflow.log_params(params)
    
    model = lgb.train(
        params,
        train_set=lgb.Dataset(X_train, y_train),
        valid_sets=[lgb.Dataset(X_val, y_val)],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50)],
    )
    
    mlflow.log_metric("val_auc", evaluate(model, X_val, y_val))
    mlflow.lightgbm.log_model(model, "model")
```

#### Day 3: Hyperparameter tuning with Optuna

```python
import optuna

def objective(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 500),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
    }
    return train_and_evaluate(params)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

Log every trial to MLflow.

#### Day 4: Evaluation + calibration

Compute and report:
- **AUC-ROC** (primary)
- **Log loss**
- **Calibration curve** (matplotlib reliability plot)
- **NDCG@10** (listwise ranking quality)
- **Precision@K, Recall@K**
- **Per-segment performance** — mobile vs desktop, new vs returning users

Apply **Platt scaling** or **isotonic regression** for calibration. Save the calibrator alongside the model.

### Deliverable

- MLflow tracking server with multiple runs visible
- Best model registered in MLflow Model Registry
- `notebooks/04_error_analysis.ipynb` showing where the model fails

---

## Phase 3: Pipeline Orchestration (3-5 days)

This is the phase that separates "Kaggle project" from "industry project". **Do not skip.**

### Tasks

#### Day 1-2: Spin up local Airflow

Use the official `docker-compose.yml` from Airflow:

```bash
curl -LO https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml
docker-compose up airflow-init
docker-compose up
```

Airflow UI at `http://localhost:8080`.

#### Day 3: Write three DAGs

**DAG 1: `feature_etl_dag.py`** — daily, idempotent
```
ingest_raw → validate_schema → compute_user_features
                              → compute_item_features      → write_to_feature_store
                              → compute_interaction_features
```

**DAG 2: `training_dag.py`** — weekly
```
fetch_training_data → train_model → evaluate → calibrate → register_if_better
```

The "register_if_better" gate is critical — it compares val AUC against current production model and only promotes if it beats by a threshold. **This is your "model validation gate" interview answer.**

**DAG 3: `monitoring_dag.py`** — daily
```
fetch_yesterday_predictions → compute_metrics → compute_drift → alert_if_threshold_breached
```

#### Day 4-5: Tie it together with Docker Compose

Create a `pipelines/docker-compose.yml` that brings up the full stack locally:
- Airflow (scheduler + webserver + worker)
- Postgres (Airflow metadata + your project DB)
- MLflow tracking server with PostgreSQL backend and MinIO artifact store
- MinIO (local S3-compatible storage)

When this works, you can demo: *"docker-compose up, navigate to localhost:8080, click Trigger DAG, watch the whole thing run"*. That's a killer demo.

### Deliverable

- Three working DAGs runnable from Airflow UI
- `docker-compose up` brings up the entire local stack
- Models registered to MLflow via the orchestrated pipeline

---

## Phase 4: Serving & API (2-3 days)

### Tasks

#### Day 1: FastAPI service

```python
# src/serving/app.py
from fastapi import FastAPI
from src.serving.predictor import Predictor

app = FastAPI(title="CTR Prediction Service", version="1.0")
predictor = Predictor.load_latest_from_mlflow()

@app.post("/predict")
async def predict(request: PredictionRequest) -> PredictionResponse:
    score = predictor.score(request.user_id, request.item_id, request.context)
    return PredictionResponse(ctr_estimate=score, model_version=predictor.version)

@app.post("/rank")
async def rank(request: RankingRequest) -> RankingResponse:
    """Score and rank a list of candidate items."""
    scores = predictor.score_batch(request.user_id, request.candidates, request.context)
    ranked = sorted(zip(request.candidates, scores), key=lambda x: -x[1])
    return RankingResponse(ranked_items=ranked[:request.top_k])

@app.get("/health")
async def health():
    return {"status": "ok", "model_version": predictor.version}
```

#### Day 2: Containerize + benchmark

```dockerfile
# docker/Dockerfile.serving
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen
COPY src/ ./src/
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Use **locust** or **vegeta** to load-test:
```bash
# Goal: p99 < 50ms for /rank with 100 candidates
vegeta attack -targets=targets.txt -rate=100/s -duration=30s | vegeta report
```

Document the latency numbers in your README — these become your "inference latency" interview number.

#### Day 3: Feature lookup at inference

In real systems, you'd query Redis for online features. For local: use a simple SQLite or in-memory dict loaded at service startup. Document the trade-off in `docs/architecture.md`.

### Deliverable

- Containerized API serving real predictions
- Latency benchmark results documented
- Postman/curl examples in README

---

## Phase 5: Monitoring & Drift Detection (2-3 days)

### Tasks

#### Day 1: Evidently for drift

```python
# src/monitoring/drift.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

def daily_drift_check(reference_df, current_df):
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(f"reports/drift_{date.today()}.html")
    
    drift_detected = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    if drift_detected:
        send_alert("Data drift detected — review reports/drift_today.html")
```

#### Day 2: Daily performance dashboard

Build a simple **Streamlit** dashboard showing:
- Daily AUC trend (last 30 days)
- Feature distribution drift (PSI per feature)
- Prediction distribution shift
- Per-segment performance breakdown

```bash
streamlit run src/monitoring/dashboard.py
```

#### Day 3: Wire into Airflow

`monitoring_dag.py` runs the drift check daily and posts results. Add a "what triggers retraining" rule:
- AUC drops > 2% rel for 3 consecutive days, OR
- PSI > 0.2 on any top-10 feature, OR
- Weekly schedule

### Deliverable

- Streamlit dashboard with screenshots in README
- Drift report HTML files committed to repo as examples
- Documented retraining triggers in `docs/architecture.md`

---

## Phase 6: A/B Test Simulation & Documentation (2-3 days)

You can't run a real A/B test in a portfolio project, so **simulate one credibly.**

### Tasks

#### Day 1: A/B test simulation

```python
# scripts/ab_test_simulation.py
"""
Simulate an A/B test by:
1. Splitting test-set users into control/treatment
2. Control: rank by item popularity (baseline)
3. Treatment: rank by ML model score
4. For each impression, decide the click using the underlying generative model
   (we have ground truth because we generated the data!)
5. Compute CTR per arm + statistical test
"""
```

This is intellectually honest — you simulate using the same generative process that created the data, so your "11% lift" (or whatever you get) is reproducible and defensible.

Compute:
- CTR per arm
- Two-proportion z-test, p-value, confidence interval
- Per-segment lift (mobile vs desktop, etc.)
- Guardrail metrics (avg dwell time per click, diversity index)

Save results to `docs/ab_test_results.md`.

#### Day 2: Documentation

**`docs/architecture.md`** — system diagram (use mermaid or excalidraw), end-to-end flow walkthrough.

**`docs/model_card.md`** — the model's "nutrition label":
- Intended use, training data, performance metrics, limitations, ethical considerations

**Updated `README.md`** with:
- Problem statement
- Architecture diagram (linked)
- Quick start (`docker-compose up`)
- Demo GIF/screenshots
- Results summary (AUC, latency, A/B test lift)
- Tech stack
- Future work

#### Day 3: Polish + record demo

- Record a 3-minute video walking through the project
- Add badges to README (CI status, license, Python version)
- Final commit-history sanity check (squash WIP commits)

### Deliverable

- Public GitHub repo, fully documented
- Demo video (Loom or YouTube unlisted)
- Architecture doc with diagram
- A/B test simulation results

---

## Resume Bullet Refresh

After completing this, your bullet becomes **bulletproof**:

> *"Built end-to-end ML pipeline (Python, LightGBM, Airflow, MLflow, FastAPI, Docker) predicting click-through rate on a content recommendation surface. Engineered point-in-time-correct features over 1.2M impressions, deployed via containerized FastAPI service with p99 latency <50ms, and demonstrated +11% CTR lift in simulated A/B test with statistical significance (p < 0.001)."*

Every claim in that bullet maps to a deliverable in this plan. No interviewer can throw a question at you that you haven't built.

---

## Recommended Sequence If You Have Limited Time

**Got 1 week?** Phases 0, 1, 2, 6 (skip orchestration, serving, monitoring). Still strong.

**Got 2-3 weeks?** Add Phase 3 (orchestration) and Phase 4 (serving). This is the sweet spot — you'll have all the interview answers covered.

**Got 4+ weeks?** Do everything. Adds Phase 5 (monitoring) which is the rarest skill in candidates and biggest differentiator.

---

## What "Industry Standard" Looks Like (Differentiators)

These are small details that signal seniority. Add them throughout:

1. **Logging and structured config** — use `loguru` or `structlog`, never `print()`. Use `hydra` or `pydantic-settings` for config.
2. **Type hints everywhere** — mypy in CI.
3. **Unit tests with fixtures** — at least one test per module. CI fails on broken tests.
4. **Pre-commit hooks** — black, ruff, mypy run before every commit.
5. **Reproducibility** — random seeds set, env locked, data hashed.
6. **Documented assumptions** — every feature pipeline has a docstring explaining the time-window logic.
7. **Idempotent pipelines** — running a DAG twice produces the same result.
8. **Schema versioning** — Pydantic models version-tagged.
9. **README that someone can follow** — clone, install, run, see results in <5 minutes.
10. **A failure mode in your write-up** — *"On new users, the model performs roughly neutral; here's why and what we'd do."* Honesty signals depth.

---

## Common Pitfalls to Avoid

- ❌ **Random train/test split** — leaks future into past, inflates metrics
- ❌ **Computing features on full dataset before splitting** — leaks val/test info into training
- ❌ **Reporting only AUC** — say nothing about calibration or business metrics
- ❌ **No baseline comparison** — "I got 0.78 AUC" is meaningless without a reference
- ❌ **Skipping containerization** — interviewers will ask "how would you deploy this?"
- ❌ **No monitoring story** — biggest single gap in junior candidates
- ❌ **Vague numbers** — always have exact dataset size, latency, lift handy

---

## Next Steps

1. Run the data generator I gave you (`generate_data.py`) — outputs everything you need
2. Set up the repo scaffold (Phase 0) — even just the empty directories
3. Pick your end-state phase target based on your timeline
4. **Start a project journal** — keep notes on every decision and trade-off. These become interview gold.

When you hit Phase 1, ping me and I'll help you write the actual EDA notebook and feature pipeline code. Same for every later phase.
