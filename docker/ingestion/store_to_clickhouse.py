from clickhouse_connect import get_client
import pandas as pd
import time

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
            client.command("SELECT 1")
            return client
        except Exception as e:
            print(f"Waiting for ClickHouse ({i+1}/10)...", e)
            time.sleep(5)

    raise RuntimeError("ClickHouse not available")

client = get_clickhouse_client()

df = pd.read_csv("/data/processed/trigger_day_processed.csv")

client.command("""
CREATE TABLE IF NOT EXISTS trigger_day_features (
    age Float32,
    amh_ng_ml Float32,
    cycle_day UInt8,
    avg_follicle_size_mm Float32,
    follicle_count UInt8,
    estradiol_pg_ml Float32,
    progesterone_ng_ml Float32,
    bmi Float32,
    basal_lh_miu_ml Float32,
    afc UInt8,
    cluster_id UInt8,
    trigger_recommended UInt8
) ENGINE = MergeTree()
ORDER BY age
""")

client.insert_df("trigger_day_features", df)

print(" Data successfully inserted into ClickHouse")
