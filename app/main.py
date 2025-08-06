from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging
from datetime import datetime
import sqlite3
from mlflow.tracking import MlflowClient

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(filename="/app/logs.txt", level=logging.INFO)

# Setup SQLite connection
conn = sqlite3.connect("/app/requests.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS MedHousePrediction (
        timestamp TEXT,
        input TEXT,
        MedHouseVal_prediction REAL
    )
''')
conn.commit()

# Load the latest model from MLflow registry
mlflow.set_tracking_uri("file:/app/mlruns")
client = MlflowClient()
model_name = "CaliforniaHousingBestModel"

latest_staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
if not latest_staging_versions:
    raise Exception(f"No model in 'Staging' stage found for '{model_name}'")

model_version = latest_staging_versions[0]
model_uri = f"file:/app/mlruns/0/{model_version.run_id}/artifacts/model"
model = mlflow.pyfunc.load_model(model_uri)
print("Loading model from:-------------", model_uri)

#model = mlflow.pyfunc.load_model("file:/app/mlruns/0/<run_id>/artifacts/model")


# Define input schema
class HousingInput(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Define predict endpoint
@app.post("/predict")
def predict(input_data: HousingInput):
    df = pd.DataFrame([input_data.dict()])
    prediction = model.predict(df)

    # Logging to file
    log = f"{datetime.now()} | Input: {df.to_dict(orient='records')} | MedHouseVal_Prediction: {prediction[0]}"
    logging.info(log)

    # Insert into SQLite
    cursor.execute(
        "INSERT INTO MedHousePrediction (timestamp, input, MedHouseVal_prediction) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), str(df.to_dict()), float(prediction[0]))
    )
    conn.commit()

    return {"MedHouseVal prediction": prediction[0]}
