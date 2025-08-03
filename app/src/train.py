import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from app.src.data_loader import load_housing_data
from app.src.pre_processing import preprocess_data
import mlflow

mlflow.set_tracking_uri("file:/app/mlruns")
# Load and preprocess data
X, y = load_housing_data()
X_train, X_test, y_train, y_test = preprocess_data(X, y)

# Define models to train
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor()
}

# Track run info
run_scores = []

# Train each model and log to MLflows
for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Log model with signature and input example
        input_example = X_train[:1]
        signature = infer_signature(X_train, preds)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        # Store run info
        run_scores.append((run.info.run_id, name, mse))

# Select best model based on MSE
best_run_id, best_model_name, best_mse = sorted(run_scores, key=lambda x: x[2])[0]
print(f"\n🏆 Best model: {best_model_name} with MSE = {best_mse:.6f}")
print(f"🔁 Registering model from run_id: {best_run_id}")

# Register best model in MLflow
client = MlflowClient()
model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "CaliforniaHousingBestModel"

# Create registry if not exists
try:
    client.create_registered_model(registered_model_name)
except:
    pass  # already exists

# Avoid duplicate version registration
existing_versions = client.search_model_versions(f"name='{registered_model_name}'")
already_registered = any(v.run_id == best_run_id for v in existing_versions)

if not already_registered:
    version = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=best_run_id
    ).version

    # Optional: add tag and promote
    client.set_model_version_tag(
        name=registered_model_name,
        version=version,
        key="description",
        value=f"Auto-registered {best_model_name} with MSE={best_mse:.6f}"
    )

    client.transition_model_version_stage(
        name=registered_model_name,
        version=version,
        stage="Staging"
    )

    print(f"✅ Model '{registered_model_name}' version {version} registered and moved to 'Staging'.")
else:
    print(f"ℹ️ Model already registered for run_id {best_run_id}. Skipping duplicate.")
