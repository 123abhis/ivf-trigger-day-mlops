import mlflow
import mlflow.sklearn
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_URI = "models:/IVF_Trigger_Day_RF@production"

model = mlflow.sklearn.load_model(MODEL_URI)

