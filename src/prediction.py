
import time
from datetime import datetime

import mlflow
import mlflow.sklearn
import pandas as pd
from clickhouse_connect import get_client


# ============================================================
# MLflow config
# ============================================================
mlflow.set_tracking_uri("http://mlflow:5000")

print("⏳ Loading production model...")
model = mlflow.sklearn.load_model(
    "models:/IVF_Trigger_Day_RF@production"
)
print("✅ Production model loaded")


# ============================================================
# ClickHouse connection
# ============================================================
def get_clickhouse_client(retries=10):
    for i in range(retries):
        try:
            client = get_client(
                host="clickhouse",
                port=8123,
                username="default",
                password="admin123",
                database="ivf_mlops"
            )
            client.query("SELECT 1")
            print("✅ Connected to ClickHouse")
            return client
        except Exception:
            print(f"⏳ Waiting for ClickHouse ({i+1}/{retries})")
            time.sleep(5)
    raise RuntimeError("❌ ClickHouse not available")


client = get_clickhouse_client()


# ============================================================
# Wait for source table
# ============================================================
def wait_for_table(client, table, retries=10):
    for i in range(retries):
        exists = client.query(
            f"EXISTS TABLE ivf_mlops.{table}"
        ).result_rows[0][0]

        if exists:
            print(f"✅ Table {table} exists")
            return

        print(f"⏳ Waiting for table {table} ({i+1}/{retries})")
        time.sleep(5)

    raise RuntimeError(f"❌ Table {table} not available")


wait_for_table(client, "trigger_day_features")


# ============================================================
# Create prediction table (NO Patient_ID)
# ============================================================
client.command("""
CREATE TABLE IF NOT EXISTS ivf_mlops.trigger_day_predictions (
    Prediction UInt8,
    Probability Float32,
    Trigger_Date DateTime
)
ENGINE = MergeTree
ORDER BY Trigger_Date
""")

print("✅ trigger_day_predictions table ready")


# ============================================================
# Fetch features
# ============================================================
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
    FROM ivf_mlops.trigger_day_features
""")

if df.empty:
    print("⚠️ No data found. Exiting.")
    exit(0)


# ============================================================
# Prepare model input
# ============================================================
X = df[model.feature_names_in_]


# ============================================================
# Predict
# ============================================================
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]


# ============================================================
# Prepare results
# ============================================================
result_df = pd.DataFrame({
    "Prediction": preds.astype("uint8"),
    "Probability": probs.astype("float32"),
    "Trigger_Date": datetime.now()
})


# ============================================================
# Insert predictions
# ============================================================
client.insert_df(
    table="trigger_day_predictions",
    df=result_df
)

print(f"✅ {len(result_df)} predictions inserted successfully")
print("🏁 Prediction job completed")
