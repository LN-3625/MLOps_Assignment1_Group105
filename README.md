# 🏠 California Housing Prediction API — MLOps End-to-End

This project is an **end-to-end MLOps implementation** using **FastAPI, MLflow, Docker, Prometheus, and Grafana** for a machine learning pipeline that predicts **Median House Value** using the California Housing dataset.

It includes:

- **Model Training & Experiment Tracking** (MLflow)
- **Model Registry & Staging**
- **Containerized API Deployment** (FastAPI + Docker)
- **Metrics & Monitoring** (Prometheus + Grafana)
- **SQLite Logging** for predictions

---

## 📂 Project Structure

```
MLOps_Assignment1_Group105/
│── app/
│   ├── data/                      # Raw data (if stored locally)
│   ├── mlruns/                    # MLflow experiment & registry storage
│   ├── src/                       # Source code
│   │   ├── data_loader.py         # Load California housing dataset
│   │   ├── pre_processing.py      # Feature preprocessing
│   │   ├── train.py               # Train models & register best one
│   │   ├── check_logs.py          # View API logs
│   ├── main.py                    # FastAPI prediction service
│   ├── requests.db                # SQLite DB for storing predictions
│   ├── app.log                    # API logs
│
│── observability/
│   ├── prometheus.yml             # Prometheus scrape configuration
│
│── .github/workflows/             # CI/CD configs
│   ├── docker-deploy.yml
│
│── docker-compose.yml             # Multi-service deployment
│── Dockerfile                     # FastAPI app image
│── requirements.txt               # Python dependencies
│── README.md                      # Documentation
```

---

## ⚙️ Tech Stack

- **FastAPI** — REST API framework
- **scikit-learn** — Model training
- **MLflow** — Experiment tracking & model registry
- **Docker + docker-compose** — Containerized deployment
- **Prometheus** — Metrics scraping
- **Grafana** — Dashboard visualization
- **SQLite** — Store prediction logs

---

## 📊 Workflow Overview

1. **Model Training**

   - Loads California Housing dataset
   - Preprocesses features
   - Trains **Linear Regression** & **Decision Tree**
   - Evaluates on test set
   - Picks best model based on **MSE**
   - Registers best model in **MLflow Model Registry** and moves it to **Staging**

2. **Model Serving**

   - FastAPI loads the latest **Staging** model from MLflow
   - Provides `/predict` endpoint
   - Logs requests into SQLite and `app.log`
   - Exposes `/metrics` endpoint for Prometheus

3. **Monitoring**
   - Prometheus scrapes `/metrics`
   - Grafana visualizes:
     - Requests/sec
     - Latency
     - Errors
     - DB writes

---

## 🖥️ Running the Project

### 1️⃣ Train and Register Model

Run locally before deployment:

```bash
# From project root
python -m app.src.train
```

This will:

- Create an MLflow experiment (`mlruns/`)
- Register the best model as `CaliforniaHousingBestModel`
- Move it to `Staging`

---

### 2️⃣ Build & Run Docker Containers

```bash
docker compose up --build
```

Services started:

- **app** — FastAPI ML API (`localhost:8000`)
- **prometheus** — Metrics scraper (`localhost:9090`)
- **grafana** — Dashboards (`localhost:3000`)

---

### 3️⃣ Test API

**Prediction request:**

```bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.9841,
  "AveBedrms": 1.0238,
  "Population": 322,
  "AveOccup": 2.5556,
  "Latitude": 37.88,
  "Longitude": -122.23
}'
```

---

### 4️⃣ Check Logs

```bash
docker compose exec app cat /app/logs/app.log
sqlite3 requests.db "SELECT * FROM MedHousePrediction;"
```

---

### 5️⃣ Prometheus Metrics

- Go to **`http://localhost:9090`**
- Example queries:

```promql
up{job="fastapi"}
predictions_total
db_writes_total
rate(predictions_total[1m])
```

---

### 6️⃣ Grafana Dashboard

- Go to **`http://localhost:3000`** (Login: `admin/admin`)
- Add Prometheus as a data source (`http://prometheus:9090`)
- Create panels for:
  - **Total Predictions**
  - **95th Percentile Latency**
  - **Database Writes**
  - **Error Rate**

---

## 📌 Key Commands Recap

```bash
# Train model
python -m app.src.train

# Start services
docker compose up --build

# Access API
http://localhost:8000/docs

# Access Prometheus
http://localhost:9090

# Access Grafana
http://localhost:3000
```
