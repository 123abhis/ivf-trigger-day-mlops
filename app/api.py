
# from fastapi import FastAPI, HTTPException
# import pandas as pd
# import time
# import logging
# from threading import Lock
# from datetime import datetime

# from app.schema import IVFInput
# from app.load_model import get_model

# # --------------------------------------------------
# # App & Logging
# # --------------------------------------------------
# app = FastAPI(title="IVF Trigger Day Prediction API")

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)s | %(message)s"
# )

# # --------------------------------------------------
# # Runtime Metrics (Thread-safe)
# # --------------------------------------------------
# metrics_lock = Lock()

# metrics = {
#     "total_requests": 0,
#     "total_errors": 0,
#     "avg_latency": 0.0,
#     "last_prediction_time": None
# }

# # --------------------------------------------------
# # Health Endpoint (Production-grade)
# # --------------------------------------------------
# @app.get("/health")
# def health_check():
#     return {
#         "status": "ok",
#         "model_loaded": model is not None,
#         "model_alias": "production",
#         "last_prediction_time": metrics["last_prediction_time"]
#     }

# # --------------------------------------------------
# # Metrics Endpoint (NEW)
# # --------------------------------------------------
# @app.get("/metrics")
# def get_metrics():
#     return metrics

# # --------------------------------------------------
# # Prediction Endpoint
# # --------------------------------------------------
# @app.post("/predict")
# def predict(data: IVFInput):
#     start_time = time.time()

#     with metrics_lock:
#         metrics["total_requests"] += 1

#     try:
#         df = pd.DataFrame([data.dict()])

#         prediction = int(model.predict(df)[0])
#         probability = float(model.predict_proba(df)[0][1])

#         latency = round(time.time() - start_time, 4)

#         with metrics_lock:
#             prev_avg = metrics["avg_latency"]
#             count = metrics["total_requests"]
#             metrics["avg_latency"] = round(
#                 ((prev_avg * (count - 1)) + latency) / count, 4
#             )
#             metrics["last_prediction_time"] = datetime.utcnow().isoformat()

#         logging.info(
#             f"Prediction={prediction} | Probability={probability} | Latency={latency}s"
#         )

#         return {
#             "prediction": prediction,
#             "probability": probability,
#             "model_alias": "production",
#             "latency_seconds": latency
#         }

#     except Exception as e:
#         with metrics_lock:
#             metrics["total_errors"] += 1

#         logging.error(f"Prediction failed: {str(e)}")
#         raise HTTPException(status_code=500, detail="Prediction failed")









from fastapi import FastAPI, HTTPException
import pandas as pd
import time
import logging
from threading import Lock
from datetime import datetime

from app.schema import IVFInput
from app.load_model import get_model   #  CHANGED (lazy load)

# --------------------------------------------------
# App & Logging
# --------------------------------------------------
app = FastAPI(title="IVF Trigger Day Prediction API")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# --------------------------------------------------
# Runtime Metrics (Thread-safe)
# --------------------------------------------------
metrics_lock = Lock()

metrics = {
    "total_requests": 0,
    "total_errors": 0,
    "avg_latency": 0.0,
    "last_prediction_time": None
}

# --------------------------------------------------
# Health Endpoint (CI + Prod safe)
# --------------------------------------------------
@app.get("/health")
def health_check():
    try:
        _ = get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_alias": "production",
        "last_prediction_time": metrics["last_prediction_time"]
    }

# --------------------------------------------------
# Metrics Endpoint
# --------------------------------------------------
@app.get("/metrics")
def get_metrics():
    return metrics

# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@app.post("/predict")
def predict(data: IVFInput):
    start_time = time.time()

    with metrics_lock:
        metrics["total_requests"] += 1

    try:
        #  Lazy model load (fixes CI)
        model = get_model()

        df = pd.DataFrame([data.dict()])

        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0][1])

        latency = round(time.time() - start_time, 4)

        with metrics_lock:
            prev_avg = metrics["avg_latency"]
            count = metrics["total_requests"]
            metrics["avg_latency"] = round(
                ((prev_avg * (count - 1)) + latency) / count, 4
            )
            metrics["last_prediction_time"] = datetime.utcnow().isoformat()

        logging.info(
            f"Prediction={prediction} | Probability={probability} | Latency={latency}s"
        )

        return {
            "prediction": prediction,
            "probability": probability,
            "model_alias": "production",
            "latency_seconds": latency
        }

    except Exception as e:
        with metrics_lock:
            metrics["total_errors"] += 1

        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
