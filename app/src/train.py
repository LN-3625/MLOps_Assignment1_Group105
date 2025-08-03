import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from app.src.data_loader import load_housing_data
from app.src.pre_processing import preprocess_data
import pandas as pd

mlflow.set_tracking_uri("file:app/mlruns")

# Load and preprocess data
X, y = load_housing_data()
X_train, X_test, y_train, y_test, feature_names = preprocess_data(X, y)
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)
# Define models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor()
}

run_scores = []

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        model.fit(X_train_df, y_train)
        preds = model.predict(X_test_df)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        # Signature fix
        input_example = X_train_df.iloc[:1]
        signature = infer_signature(X_train_df, preds)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example
        )

        run_scores.append((run.info.run_id, name, mse))

# Pick best model
best_run_id, best_model_name, best_mse = sorted(run_scores, key=lambda x: x[2])[0]
print(f"\n🏆 Best model: {best_model_name} with MSE = {best_mse:.6f}")
print(f"🔁 Registering model from run_id: {best_run_id}")

# Register in MLflow
client = MlflowClient()
model_uri = f"runs:/{best_run_id}/model"
registered_model_name = "CaliforniaHousingBestModel"

try:
    client.create_registered_model(registered_model_name)
except:
    pass

existing_versions = client.search_model_versions(f"name='{registered_model_name}'")
already_registered = any(v.run_id == best_run_id for v in existing_versions)

if not already_registered:
    version = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=best_run_id
    ).version

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
