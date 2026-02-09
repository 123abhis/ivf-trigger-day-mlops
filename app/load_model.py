# import mlflow
# import mlflow.sklearn
# import os

# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# MODEL_URI = "models:/IVF_Trigger_Day_RF@production"

# model = mlflow.sklearn.load_model(MODEL_URI)




# app/load_model.py
import mlflow
import mlflow.sklearn
import os

MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/IVF_Trigger_Day_RF@production"
)

_model = None

def get_model():
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model(MODEL_URI)
    return _model
