from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

# --- FastAPI App ---
app = FastAPI()
from prometheus_fastapi_instrumentator import Instrumentator

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)
# --- Logging ---
os.makedirs("/app/logs", exist_ok=True)
logging.basicConfig(filename="/app/logs/app.log", level=logging.INFO)

# --- SQLite DB ---
conn = sqlite3.connect("requests.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS MedHousePrediction (
        timestamp TEXT,
        input TEXT,
        MedHouseVal_prediction REAL
    )
""")
conn.commit()

# --- Prometheus Metrics ---
PREDICTIONS_TOTAL = Counter(
    "predictions_total",
    "Number of /predict calls",
    ["status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of /predict endpoint",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)
DB_WRITES_TOTAL = Counter(
    "db_writes_total",
    "Number of rows inserted into SQLite for predictions"
)

# --- MLflow Setup (Experiment ID 0 only) ---
mlruns_path = Path("/app/mlruns").resolve()
store_uri = f"file:///{mlruns_path.as_posix()}"

mlflow.set_tracking_uri(store_uri)
mlflow.set_registry_uri(store_uri)

client = MlflowClient()
model_name = "CaliforniaHousingBestModel"
latest_staging_versions = client.get_latest_versions(model_name, stages=["Staging"])

if not latest_staging_versions:
    raise Exception(f"No model in 'Staging' stage found for '{model_name}'")

model_version = latest_staging_versions[0]

# Force experiment ID 0
model_uri = f"{store_uri}/0/{model_version.run_id}/artifacts/model"
model = mlflow.pyfunc.load_model(model_uri)
print(f"✅ Loading model from: {model_uri}")

# --- Input Schema ---
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# --- Predict Endpoint ---
@app.post("/predict")
def predict(input_data: HousingInput):
    import time
    start_time = time.time()

    try:
        df = pd.DataFrame([input_data.dict()])
        prediction = model.predict(df)

        logging.info(f"{datetime.now()} | Input: {df.to_dict(orient='records')} | Prediction: {prediction[0]}")
        
        cursor.execute(
            "INSERT INTO MedHousePrediction (timestamp, input, MedHouseVal_prediction) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), str(df.to_dict()), float(prediction[0]))
        )
        conn.commit()
        DB_WRITES_TOTAL.inc()

        PREDICTIONS_TOTAL.labels(status="ok").inc()
        return {"MedHouseVal prediction": prediction[0]}

    except Exception as e:
        PREDICTIONS_TOTAL.labels(status="error").inc()
        return {"error": str(e)}

    finally:
        PREDICTION_LATENCY.observe(time.time() - start_time)

# --- Expose Prometheus /metrics ---
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
