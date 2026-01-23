import pandas as pd
import clickhouse_connect
from datetime import datetime

client = clickhouse_connect.get_client(
    host="localhost",
    port=8123,
    username="default",
    password="admin123",
    database="ivf_mlops"
)

df = pd.read_csv("data/processed/trigger_day_processed.csv")

# Safety check
print("CSV Columns:", list(df.columns))

# Add ingestion timestamp
df["ingestion_time"] = datetime.now()

client.command("""
CREATE TABLE IF NOT EXISTS trigger_day_features (
    age Float32,
    amh_ng_ml Float32,
    cycle_day Int32,
    avg_follicle_size_mm Float32,
    follicle_count Int32,
    estradiol_pg_ml Float32,
    progesterone_ng_ml Float32,
    bmi Float32,
    basal_lh_miu_ml Float32,
    afc Int32,
    cluster_id Int32,
    trigger_recommended UInt8,
    ingestion_time DateTime
) ENGINE = MergeTree()
ORDER BY ingestion_time
""")

client.insert(
    table="trigger_day_features",
    data=df.values.tolist(),
    column_names=df.columns.tolist()
)

print("✅ Processed + clustered data stored in ClickHouse")
