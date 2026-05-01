# CTR Prediction — Production-Grade MLOps Build Plan (8 Weeks)

> **Goal:** Build a production-grade ML system end-to-end on AWS, with the depth of a real MLOps engineering job. Every "industry standard" tool in your toolkit by the end.
>
> **Timeline:** 8 weeks (1-2 months committed)
> **Budget:** AWS Free Tier (Free Plan: $200 credits + Always-Free services for 6 months)
> **Outcome:** A live, public, observable ML system + a portfolio that gets you past most senior ML/MLOps screening rounds.

---

## Important: AWS Free Tier — How It Actually Works in 2026

If you signed up for AWS **before July 15, 2025**, you have the legacy 12-month free tier. If you sign up **now**, you get the new **Free Plan**:

- **$100 credits** on signup + **$100 more** by completing onboarding activities = **$200 total**
- **6 months** OR until credits run out — whichever comes first
- **Always-Free services** (no time limit): Lambda (1M req/mo), DynamoDB (25GB), S3 (5GB), CloudFront (1TB egress), SNS (1M publishes), 10 CloudWatch metrics + alarms
- **EC2 t3.micro: 750 hours/month always-free** (one instance running 24/7 fits)
- **RDS db.t3.micro: 750 hours/month** (12-month tier)

**For an 8-week project, $200 in credits + always-free services is plenty** — you'll likely use $30-50 in credits if you follow this plan and clean up after yourself.

### Cost discipline rules (set these on Day 1)

1. **Set a $5 zero-spend budget alert** in AWS Budgets. Email when forecasted to exceed.
2. **Set a hard $20 budget limit** with auto-shutdown via SNS → Lambda. (We'll wire this in Phase 0.)
3. **Use ONE region for everything** (`ap-south-1` Mumbai is closest to you; `us-east-1` has cheapest pricing globally). Free Tier hours are *shared across regions* — running an instance in two regions burns the budget twice as fast.
4. **Tag every resource** with `Project=ctr-prediction` so you can clean up easily.
5. **One running EC2 instance maximum.** If you need a second box, stop the first.
6. **Daily 60-second cleanup check** — covered in the runbook below.

### The "trap" services to avoid

| Service | Why it bites |
|---|---|
| **NAT Gateway** | $0.045/hr ($32/mo) and **not in free tier**. Use VPC endpoints or a NAT instance instead. |
| **Elastic IP** unattached | $3.60/mo per unused EIP. Always release unused. |
| **EBS volumes after instance termination** | Persist and bill until deleted. |
| **CloudWatch Logs** | First 5GB free, then $0.50/GB. Log rotation matters. |
| **Managed services on the "no" list** | MSK (Kafka), Managed Airflow (MWAA), SageMaker — all blow the budget fast. |

---

## The Stack (Cloud-Native, Industry-Standard)

This is the stack — every tool here is what real teams use in production. Each one signals seniority in interviews.

### Core toolchain

| Layer | Tool | Free-tier strategy |
|---|---|---|
| **IaC** | Terraform | Free. State stored in S3 backend. |
| **Source control + CI/CD** | GitHub + GitHub Actions | Free for public repos, 2000 min/mo for private. |
| **Compute (services)** | EC2 t3.micro running Docker Compose | Always-free 750 hrs/mo. |
| **Compute (training)** | EC2 t3.medium spot, on-demand | ~$0.01/hr spot. Stop after run. |
| **Object storage** | S3 | 5GB always free. |
| **Database** | RDS PostgreSQL db.t3.micro | 750 hrs/mo, 20GB storage (12-month tier). |
| **Container registry** | ECR | 500MB always free. |
| **Orchestration** | Self-hosted Airflow (Docker on EC2) | Free. Avoids MWAA ($300/mo). |
| **Experiment tracking** | Self-hosted MLflow | Free. Backed by RDS + S3. |
| **Feature store** | Feast (open-source) | Free. Offline = S3, online = DynamoDB (always-free). |
| **Modeling** | LightGBM / XGBoost | Free. |
| **Serving** | FastAPI in Docker on EC2, behind ALB | ALB has free tier; if too expensive, use Caddy reverse proxy. |
| **Monitoring (drift)** | Evidently AI | Free open-source. |
| **Monitoring (system)** | Prometheus + Grafana on EC2, OR CloudWatch | Both free at our scale. |
| **Logs** | CloudWatch Logs | 5GB free, then careful. |
| **Secrets** | AWS Secrets Manager (one secret) OR `.env` + SSM Parameter Store | SSM Parameter Store is free for standard params. |
| **DNS (optional)** | Route 53 | $0.50/mo per hosted zone. Skip if not needed. |
| **Notifications** | SNS → Slack/email | 1M publishes free. |

### Deliberate omissions and why

- **No Kubernetes / EKS.** EKS control plane is $0.10/hr = $73/month. Not free-tier eligible. EC2 + Docker Compose teaches the same concepts with zero infra cost. Mention K8s in your README as "future migration path."
- **No SageMaker.** Easy to blow the budget; doesn't teach you anything you can't learn with EC2 + MLflow.
- **No MWAA (Managed Airflow).** $300+/month minimum.
- **No Redshift / EMR.** Not free-tier-friendly.
- **No managed Kafka (MSK).** Use self-hosted Kafka in Docker if you need streaming, OR skip streaming entirely and use S3 + scheduled jobs.

---

## Architecture (What You're Building)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              GitHub                                      │
│  Source code, CI/CD workflows, Terraform configs, model versioning       │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │ git push / PR
                      ▼
            ┌─────────────────────┐
            │  GitHub Actions      │
            │  - Lint, test        │
            │  - Build Docker img  │
            │  - Push to ECR       │
            │  - terraform plan    │
            └──────────┬───────────┘
                       │
           ┌───────────▼────────────────────────┐
           │          AWS Account               │
           │                                     │
           │  ┌──────────────────────────────┐ │
           │  │    EC2 (t3.micro/small)      │ │
           │  │   Docker Compose runs:        │ │
           │  │   - Airflow (scheduler+web)   │ │
           │  │   - MLflow tracking server    │ │
           │  │   - FastAPI serving           │ │
           │  │   - Grafana                   │ │
           │  │   - Evidently dashboard       │ │
           │  └──────┬───────────────────────┘ │
           │         │                            │
           │         ├─→ S3 (raw data, features,  │
           │         │       MLflow artifacts,    │
           │         │       Terraform state)     │
           │         │                            │
           │         ├─→ RDS Postgres             │
           │         │   (Airflow metadata,       │
           │         │    MLflow backend,         │
           │         │    feature offline store)  │
           │         │                            │
           │         ├─→ DynamoDB (online         │
           │         │     feature store)         │
           │         │                            │
           │         └─→ ECR (Docker images)      │
           │                                     │
           │  ┌──────────────────────────────┐  │
           │  │ CloudWatch (logs, metrics,    │ │
           │  │ alarms) + SNS (alerts)        │ │
           │  └──────────────────────────────┘ │
           └────────────────────────────────────┘
```

This is a real production architecture in miniature. Every box on this diagram is a talking point in an interview.

---

## 8-Week Phase Plan

Each phase ends with: **a deliverable, a test, and a git tag.** No hand-waving — you should be able to demo each milestone.

### Week 1 — Foundation & Cloud Setup

**Phase 0a: Local scaffolding (Days 1-2)**

- Initialize repo with the structure from the previous plan doc (already covered).
- Set up `uv` env, pyproject.toml, pre-commit hooks (black, ruff, mypy).
- Run the data generator (`scripts/generate_data.py`) — produces ~6M impressions locally.
- Write the GitHub Actions CI workflow for lint + test on every PR.
- **Tag:** `v0.1-scaffold`

**Phase 0b: AWS account & cost guardrails (Day 3)**

This is the most important day of the whole project. Get this wrong and you'll wake up to a surprise bill.

1. Sign up for AWS Free Plan. Verify $100 credit appears.
2. **Set MFA on root account.** Create an IAM admin user; never use root again.
3. Install AWS CLI, configure `aws configure --profile ctr-project`.
4. **Create budgets** (do this BEFORE provisioning anything):
   ```bash
   # $5 forecasted spend alert
   aws budgets create-budget --account-id <id> --budget file://budget-5usd.json
   # $20 hard limit with SNS → Lambda auto-shutdown
   aws budgets create-budget --account-id <id> --budget file://budget-20usd-hard.json
   ```
5. Enable **AWS Cost Anomaly Detection** (free).
6. Enable Free Tier usage alerts in Billing Preferences.
7. Pick **one region** (`ap-south-1` or `us-east-1`) and stick with it.

**Daily cleanup check** — add this to your terminal alias:
```bash
alias aws-check='aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query "Reservations[].Instances[].{ID:InstanceId,Type:InstanceType,Launch:LaunchTime}" --output table && \
aws ec2 describe-volumes --filters "Name=status,Values=available" --query "Volumes[].{ID:VolumeId,Size:Size}" --output table && \
aws ec2 describe-addresses --query "Addresses[?AssociationId==null]" --output table'
```

**Tag:** `v0.2-aws-setup`

**Phase 0c: Terraform foundation (Days 4-5)**

Provision the base infrastructure with Terraform — never click in the console for anything that should be IaC.

```
terraform/
├── backend.tf              # S3 backend for state
├── providers.tf
├── variables.tf
├── modules/
│   ├── networking/         # VPC, subnets, security groups
│   ├── storage/            # S3 buckets, ECR repos
│   ├── database/           # RDS instance
│   ├── compute/            # EC2 instance, IAM roles
│   └── monitoring/         # CloudWatch, SNS, budgets
├── environments/
│   ├── dev/
│   └── prod/
└── README.md
```

What to provision in Week 1:
- VPC with public + private subnets, internet gateway, **no NAT gateway** (avoid the $32/mo charge — use VPC endpoints if private subnet needs S3).
- S3 buckets: `ctr-data-raw`, `ctr-data-processed`, `ctr-mlflow-artifacts`, `ctr-terraform-state`
- ECR repositories: `ctr/airflow`, `ctr/mlflow`, `ctr/serving`, `ctr/training`
- IAM roles for EC2 instance with least-privilege S3 access
- Security groups (SSH from your IP only, HTTPS open, internal traffic between services)

**Don't provision EC2 or RDS yet** — they cost hours, and you don't need them running until Week 3.

**Test:**
```bash
terraform plan   # shows expected resources
terraform apply  # provisions
aws s3 ls         # verify buckets exist
```

**Tag:** `v0.3-terraform-foundation`

---

### Week 2 — EDA, Features, Baseline Model

**Phase 1a: EDA + data validation (Days 1-3)**

- `notebooks/01_eda.ipynb` — full exploratory analysis
- Schema definitions in `src/data/validate.py` using **Pandera**
- CI pipeline runs schema validation on test data

Key analyses to document:

| Question | Output |
|---|---|
| What's the overall CTR? | One number + confidence interval |
| CTR by category, device, position, hour | Bar charts, saved as PNGs |
| Distribution of impressions/user (long tail) | Log-scale histogram |
| Missing data audit | Table with % missing per column |
| Time-of-day and day-of-week patterns | Heatmap |
| Are there bots/outlier users? | Identify users with > 99th percentile impressions/day |

Save all charts to `reports/figures/` and reference them in `docs/eda_summary.md`.

**Phase 1b: Feature engineering (Days 4-5)**

Implement features in three modules — `user_features.py`, `item_features.py`, `interaction_features.py`. **Every aggregation must be point-in-time correct.**

Write unit tests for the leakage check:
```python
def test_user_ctr_no_leakage():
    """Feature value at impression t should only use data BEFORE t."""
    impressions = generate_test_data()
    feature = compute_user_ctr_7d(impressions, target_timestamp=T)
    assert all(impressions[impressions.user_id == u].timestamp < T 
               for u in feature.index)
```

**Test:** End-to-end pipeline `make features` produces `data/processed/training_data.parquet` and passes Pandera validation.

**Tag:** `v1.0-features`

---

### Week 3 — Modeling, MLflow, Cloud Deployment

**Phase 2a: Local modeling + MLflow (Days 1-3)**

- Set up local MLflow tracking (`mlflow ui` running locally for now)
- Implement `src/models/train.py`: time-based split, LightGBM, log everything to MLflow
- Run hyperparameter tuning with **Optuna** (~50 trials)
- Implement Platt scaling calibration in `src/models/calibrate.py`
- Run error analysis notebook: per-segment AUC, calibration plots, SHAP feature importance

**Phase 2b: Deploy MLflow + RDS to AWS (Days 4-5)**

This is where the cloud journey starts. Move MLflow from local to AWS so all experiments are tracked centrally.

1. **Provision RDS via Terraform:**
   ```hcl
   module "rds" {
     source            = "./modules/database"
     instance_class    = "db.t3.micro"  # free tier
     allocated_storage = 20             # free tier max
     engine            = "postgres"
     # ... 
   }
   ```
2. **Provision EC2 (t3.micro)** with Docker installed. User-data script bootstraps Docker Compose.
3. **Run MLflow on EC2:**
   ```yaml
   # docker-compose.yml
   services:
     mlflow:
       image: ghcr.io/mlflow/mlflow:latest
       ports: ["5000:5000"]
       command: >
         mlflow server
         --backend-store-uri postgresql://user:pass@${RDS_HOST}/mlflow
         --default-artifact-root s3://ctr-mlflow-artifacts/
         --host 0.0.0.0
   ```
4. **Configure local MLflow client to point at the cloud server.** Re-run training, see runs appear in the cloud UI.

**Test:** Open `http://<ec2-ip>:5000` (locked down to your IP), see your training runs.

**Tag:** `v2.0-mlflow-cloud`

---

### Week 4 — Airflow Orchestration

This is the real "MLOps" week. Pay attention here — orchestration is the most-asked-about and least-understood topic in MLOps interviews.

**Phase 3a: Self-hosted Airflow on EC2 (Days 1-2)**

Add Airflow to the same `docker-compose.yml`:

```yaml
services:
  airflow-scheduler:
    image: apache/airflow:2.10.0
    depends_on: [postgres]
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://...
    volumes:
      - ./dags:/opt/airflow/dags
    command: scheduler
  airflow-webserver:
    # ... port 8080
```

**Critical:** EC2 t3.micro has **1 GB RAM** — running Airflow + MLflow + Postgres + FastAPI on it is tight. You may need to upgrade to **t3.small (2 GB RAM)** at $0.018/hr (~$13/mo). Budget for this. Alternative: split services across two t3.micros (still both free).

**Phase 3b: Three production DAGs (Days 3-5)**

**DAG 1: `feature_etl_dag.py`** (daily, idempotent)

```python
@dag(schedule="@daily", catchup=True, max_active_runs=1)
def feature_etl():
    raw = ingest_from_s3()        # PythonOperator
    validated = validate(raw)      # Pandera schema check
    user_feats = compute_user_features(validated)
    item_feats = compute_item_features(validated)
    interaction_feats = compute_interaction_features(validated, user_feats, item_feats)
    write_to_feature_store([user_feats, item_feats, interaction_feats])
```

Use **TaskFlow API** (modern Airflow). Use **XComs** sparingly — pass S3 paths between tasks, not data.

**DAG 2: `training_dag.py`** (weekly + manual trigger)

```
fetch_features → train_model → evaluate_model → 
    [if val_auc > prod_auc + 0.002] register_to_mlflow → notify_slack
```

The "register only if better" gate is your **model validation gate** — a key interview answer.

**DAG 3: `monitoring_dag.py`** (daily)

```
fetch_yesterday_predictions → compute_actual_metrics → 
    compare_to_training → drift_report → alert_if_breach
```

**Test:** Trigger each DAG from the Airflow UI; watch them complete green.

**Tag:** `v3.0-airflow`

---

### Week 5 — Feature Store with Feast

**Phase 4a: Feast offline store (Days 1-2)**

Feast is the production feature store standard. Set it up properly — interviewers love when candidates can articulate online vs offline feature retrieval and point-in-time joins.

```bash
feast init ctr_features
```

Define feature views:
```python
# feature_repo/user_features.py
user_stats_view = FeatureView(
    name="user_stats",
    entities=[user_entity],
    ttl=timedelta(days=30),
    schema=[
        Field(name="user_ctr_7d", dtype=Float32),
        Field(name="user_session_count_30d", dtype=Int32),
    ],
    source=FileSource(
        path="s3://ctr-data-processed/user_features.parquet",
        timestamp_field="event_timestamp",
    ),
    online=True,
)
```

**Phase 4b: Online store on DynamoDB (Days 3-4)**

DynamoDB is in the always-free tier — perfect for the online feature store.

```yaml
# feature_store.yaml
project: ctr
provider: aws
registry: s3://ctr-data-processed/feast-registry.db
online_store:
  type: dynamodb
  region: ap-south-1
offline_store:
  type: file
```

`feast materialize` pushes features from offline → online store. Wire this into your training DAG.

**Phase 4c: Point-in-time correctness test (Day 5)**

This is the killer demo. Show that:
1. At training time, features are computed as-of impression time (no future leakage).
2. At inference time, online store returns the latest available feature values.

Write a test that proves this. Save to `tests/test_point_in_time.py` — make sure it's in your CI pipeline.

**Tag:** `v4.0-feast`

---

### Week 6 — Serving + ECR + Auto-deployment

**Phase 5a: FastAPI serving (Days 1-2)**

```python
# src/serving/app.py
from fastapi import FastAPI
from src.serving.predictor import Predictor

app = FastAPI(title="CTR Service", version=os.getenv("MODEL_VERSION"))
predictor = Predictor.load_from_mlflow(stage="Production")
feature_store = FeatureStore(repo_path="feature_repo/")

@app.post("/rank")
async def rank(request: RankingRequest):
    # 1. Fetch online features for user + candidates
    features = feature_store.get_online_features(
        features=["user_stats:user_ctr_7d", "item_stats:item_ctr_7d", ...],
        entity_rows=[{"user_id": request.user_id, "item_id": iid} 
                     for iid in request.candidates]
    ).to_df()
    # 2. Score
    scores = predictor.predict(features)
    # 3. Rank and return
    ranked = sorted(zip(request.candidates, scores), key=lambda x: -x[1])
    return {"ranked_items": ranked[:request.top_k], "model_version": predictor.version}
```

**Phase 5b: Containerize and push to ECR (Day 3)**

`docker/Dockerfile.serving` — use multi-stage builds, slim base image, non-root user. CI auto-builds on every commit to `main` and pushes to ECR.

```yaml
# .github/workflows/deploy.yml
- name: Build and push to ECR
  run: |
    aws ecr get-login-password | docker login --username AWS --password-stdin $ECR_URI
    docker build -t $ECR_URI/ctr/serving:${{ github.sha }} -f docker/Dockerfile.serving .
    docker push $ECR_URI/ctr/serving:${{ github.sha }}
```

**Phase 5c: Deploy to EC2 + benchmark (Days 4-5)**

Add the serving container to your `docker-compose.yml`. Use **Watchtower** (free, open source) or a simple cron-based git pull to auto-update when a new image is pushed to ECR.

Load test with **Locust:**
```bash
locust -f tests/load_test.py --host=http://<ec2-ip>:8000 --users=50 --spawn-rate=5
```

Document your latency results (p50, p95, p99) in `docs/serving_benchmarks.md`. **Goal: p99 < 100ms** on a t3.micro for /rank with 50 candidates. If you hit it, that's your interview number.

**Tag:** `v5.0-serving`

---

### Week 7 — Monitoring, Drift, Observability

**Phase 6a: Prometheus + Grafana (Days 1-2)**

Add to docker-compose:
```yaml
services:
  prometheus:
    image: prom/prometheus
    ports: ["9090:9090"]
    volumes: ["./prometheus.yml:/etc/prometheus/prometheus.yml"]
  grafana:
    image: grafana/grafana
    ports: ["3000:3000"]
```

Instrument FastAPI with `prometheus-fastapi-instrumentator`:
```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Build Grafana dashboards for:
- Request rate, error rate, p50/p95/p99 latency
- Predicted score distribution over time
- Active model version
- Feature lookup latency (Feast → DynamoDB)

**Phase 6b: Evidently for data drift (Days 3-4)**

```python
# src/monitoring/drift.py
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

def daily_drift_check():
    reference = load_training_distribution()
    current = load_yesterday_serving_data()
    
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=reference, current_data=current)
    
    # Save HTML to S3 for archival
    report.save_html(f"/tmp/drift_{date.today()}.html")
    s3.upload_file(...)
    
    if report.as_dict()["metrics"][0]["result"]["dataset_drift"]:
        send_slack_alert("Drift detected — see S3 link")
```

Wire this into `monitoring_dag.py`. Drift reports archived to S3 daily.

**Phase 6c: CloudWatch alarms + SNS (Day 5)**

Define alarms in Terraform:
- API error rate > 1% for 5 min → page (SNS → email)
- p99 latency > 200ms for 10 min → warn
- Daily AUC drops > 5% from rolling 30-day avg → critical

**Test:** Manually inject a drift signal (push a new column distribution); confirm alert fires.

**Tag:** `v6.0-observability`

---

### Week 8 — A/B Testing, CI/CD Polish, Documentation

**Phase 7a: A/B test simulation (Days 1-2)**

Since you generated the data, you have the ground-truth click probability function. Use it to simulate an A/B test honestly:

```python
# scripts/ab_test_simulation.py
def simulate_ab_test():
    test_users = load_test_users()
    control_users, treatment_users = stratified_split(test_users)
    
    for user in control_users:
        candidates = candidate_generator(user)
        ranked = rank_by_popularity(candidates)  # baseline
        impressions = serve_top_k(ranked, k=10)
        clicks = simulate_clicks_with_ground_truth(user, impressions)
        log_outcomes("control", impressions, clicks)
    
    for user in treatment_users:
        candidates = candidate_generator(user)
        ranked = rank_by_model(candidates, user)  # ML model
        impressions = serve_top_k(ranked, k=10)
        clicks = simulate_clicks_with_ground_truth(user, impressions)
        log_outcomes("treatment", impressions, clicks)
    
    # Two-proportion z-test, confidence interval, segment analysis
    report = generate_ab_test_report()
```

Document in `docs/ab_test_results.md`:
- Headline lift (CTR_treatment / CTR_control - 1)
- Statistical significance (p-value, CI)
- Per-segment lift (mobile/desktop, new/returning users)
- Guardrail metric impacts (avg dwell, diversity, latency)
- Honest write-up of where the model performed worse

**Phase 7b: CI/CD hardening (Days 3-4)**

Beef up GitHub Actions:

| Workflow | Trigger | Action |
|---|---|---|
| `ci.yml` | Every PR | Lint, type-check, run unit tests, validate Terraform plan |
| `deploy-infra.yml` | Push to main + path filter on `terraform/` | terraform apply (with manual approval) |
| `deploy-serving.yml` | Push to main + path filter on `src/serving/**` | Build + push image to ECR, restart container |
| `train.yml` | Manual + weekly cron | Trigger training DAG via Airflow API |
| `cost-check.yml` | Daily | Check Free Tier usage, fail if > 80% on any service |

**Phase 7c: Documentation + portfolio polish (Day 5)**

The single most important deliverable. Most candidates have decent code and zero documentation. Reverse this.

Create:

1. **`README.md`** — the front door. Includes:
   - 1-paragraph problem statement
   - **Architecture diagram** (use draw.io, mermaid, or excalidraw — embed as PNG in repo)
   - **Live demo links** (if you keep services running) or screenshots
   - Quick start: `make local-up` brings up local docker-compose for anyone to try
   - Tech stack list
   - Results summary (AUC, latency, A/B lift)
   - Cost summary ("Total AWS spend: $X over 8 weeks")
   - Future work

2. **`docs/architecture.md`** — deep dive. Component-by-component walkthrough, data flow, failure modes.

3. **`docs/model_card.md`** — model "spec sheet" (Google's model card format).

4. **`docs/runbook.md`** — what to do when things break. Real ops document.

5. **`docs/decisions/`** — ADRs (Architecture Decision Records) for major choices: "Why LightGBM over XGBoost", "Why self-hosted Airflow over MWAA", "Why DynamoDB for online features", etc. **This is what senior candidates produce.**

6. **Demo video** — 5-minute Loom walkthrough. Show:
   - Architecture diagram (30s)
   - Trigger a training DAG, watch it complete (1m)
   - Show MLflow runs (30s)
   - Show Grafana dashboard with live metrics (1m)
   - Hit the API with curl, get a ranked response (30s)
   - Show A/B test results (1m)
   - Wrap (30s)

**Tag:** `v1.0-release` 🎉

---

## End-of-Project Cleanup (Critical)

Before your AWS Free Plan ends or you stop actively using the project:

```bash
# Save your work first
git push --tags
docker save -o ctr-serving.tar ctr/serving:latest

# Tear down expensive resources but keep data
terraform destroy -target=module.compute  # kills EC2
terraform destroy -target=module.database # kills RDS

# OR full teardown
terraform destroy

# Verify nothing's still running
aws-check  # the alias from Phase 0b
```

S3 buckets with your data and code can stay — 5GB is free forever.

---

## Resume Bullet (Final Form)

After completing this:

> **MLOps Pipeline — CTR Prediction System (AWS, Airflow, MLflow, Feast)**
> Designed and deployed end-to-end production ML system on AWS predicting click-through rate, serving ranked recommendations via FastAPI with p99 latency under 100ms. Built with Terraform (IaC), GitHub Actions (CI/CD), self-hosted Airflow + MLflow for orchestration and tracking, Feast feature store with DynamoDB online layer, Evidently + Prometheus + Grafana for drift and operational monitoring. Demonstrated +11% CTR lift in simulated A/B test (p < 0.001). Project, architecture, and runbook publicly documented at github.com/yourname/ctr-prediction.

That bullet **is** the senior MLOps job description. Every word maps to a deliverable.

---

## Weekly Checkpoint Schedule

End of every week, write a short checkpoint in `docs/journal/week-N.md`:

- What did I build?
- What broke?
- What's the AWS cost so far?
- What did I learn that I didn't know on Monday?
- What's blocking me?

These are gold for interviews — you'll get asked "tell me about a time something went wrong" and you'll have receipts.

---

## What to Do When You Get Stuck

In rough order of effectiveness:

1. **Read the official docs** — Airflow, MLflow, Feast, Terraform docs are all excellent.
2. **Search the project's GitHub issues** — someone has hit your problem before.
3. **Stack Overflow with the exact error message in quotes.**
4. **Ask me** — paste the error + what you tried, I can usually unblock.
5. **Step away for an hour.** No joke. Half of debugging is breaking out of mental ruts.

---

## What I'll Help You With Next (Just Ask)

When you're ready for any phase, ping me with the phase number and I'll generate:

- Full code for the relevant module(s)
- Terraform configs for the AWS pieces
- The actual EDA notebook (not just an outline)
- Test cases
- Debugging help when things break

This plan is the map. The code is the territory. Take it one phase at a time, commit often, and you'll have something special at the end.
