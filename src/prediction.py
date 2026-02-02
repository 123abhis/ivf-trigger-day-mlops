# import mlflow
# import mlflow.sklearn
# from clickhouse_connect import get_client
# import pandas as pd
# import time
# from datetime import datetime

# mlflow.set_tracking_uri("http://mlflow:5000")


# # Load model
# mlflow.set_tracking_uri("http://mlflow:5000")

# model = mlflow.sklearn.load_model(
#     "models:/IVF_Trigger_Day_RF@challenger"
# )


# def get_clickhouse_client():
#     for _ in range(10):
#         try:
#             client = get_client(
#                 host="clickhouse",
#                 port=8123,
#                 username="default",
#                 password="admin123",
#                 database="ivf_mlops"
#             )
#             client.query("SELECT 1")
#             return client
#         except Exception:
#             time.sleep(5)
#     raise RuntimeError("ClickHouse not available")

# client = get_clickhouse_client()

# df = client.query_df("""
#     SELECT
#         patient_id,
#         age,
#         amh_ng_ml,
#         cycle_day,
#         avg_follicle_size_mm,
#         follicle_count,
#         estradiol_pg_ml,
#         progesterone_ng_ml,
#         bmi,
#         basal_lh_miu_ml,
#         afc,
#         cluster_id
#     FROM trigger_day_features
# """)

# if df.empty:
#     print("No data available for prediction")
#     exit(0)

# patient_ids = df["patient_id"]
# df = df.drop(columns=["patient_id"])

# # Ensure feature order
# df = df[model.feature_names_in_]

# predictions = model.predict(df)
# probabilities = model.predict_proba(df)[:, 1]

# result_df = pd.DataFrame({
#     "patient_id": patient_ids,
#     "prediction": predictions,
#     "probability": probabilities,
#     "model_alias": "Challenger",
#     "prediction_time": datetime.now()
# })

# client.insert_df(
#     table="trigger_day_predictions",
#     df=result_df
# )

# with mlflow.start_run(run_name="rf_batch_prediction", nested=True):
#     mlflow.log_param("model_alias", "Challenger")
#     mlflow.log_metric("batch_size", len(df))
#     mlflow.log_metric("positive_rate", predictions.mean())

# print(" Batch prediction completed successfully")

import mlflow
import mlflow.sklearn
from clickhouse_connect import get_client
import pandas as pd
import time
from datetime import datetime

# ---------------- MLflow ----------------
mlflow.set_tracking_uri("http://mlflow:5000")


def wait_for_mlflow():
    for i in range(10):
        try:
            mlflow.search_experiments()
            return
        except Exception:
            print(f"Waiting for MLflow... ({i+1}/10)")
            time.sleep(5)
    raise RuntimeError("MLflow not available")


wait_for_mlflow()

# Load challenger model by alias
# model = mlflow.sklearn.load_model(
#     model_uri="models:/IVF_Trigger_Day_RF@challenger"
# )

# Load production model by alias
model = mlflow.sklearn.load_model(
    model_uri="models:/IVF_Trigger_Day_RF@production"
)



# ---------------- ClickHouse ----------------
def get_clickhouse_client():
    for i in range(10):
        try:
            client = get_client(
                host="clickhouse",
                port=8123,
                username="default",
                password="admin123",
                database="ivf_mlops"
            )
            client.query("SELECT 1")
            return client
        except Exception:
            print(f"Waiting for ClickHouse... ({i+1}/10)")
            time.sleep(5)
    raise RuntimeError("ClickHouse not available")


client = get_clickhouse_client()

# ✅ EXACT same features as training
df = client.query_df("""
    SELECT
        age,
        amh_ng_ml,
        cycle_day,
        avg_follicle_size_mm,
        follicle_count,
        estradiol_pg_ml,
        progesterone_ng_ml,
        bmi,
        basal_lh_miu_ml,
        afc,
        cluster_id
    FROM trigger_day_features
""")

if df.empty:
    print("No data available for prediction")
    exit(0)

# Ensure correct feature order
df = df[model.feature_names_in_]

# Predict
predictions = model.predict(df)
probabilities = model.predict_proba(df)[:, 1]

# Result dataframe
result_df = pd.DataFrame({
    "prediction": predictions,
    "probability": probabilities,
    "model_alias": "production",
    "prediction_time": datetime.now()
})

# Insert predictions
client.insert_df(
    table="trigger_day_predictions",
    df=result_df
)

# Log batch prediction run
with mlflow.start_run(run_name="rf_batch_prediction"):
    mlflow.log_param("model_alias", "production")
    mlflow.log_metric("batch_size", len(df))
    mlflow.log_metric("positive_rate", predictions.mean())

print("✅ Batch prediction completed successfully")
