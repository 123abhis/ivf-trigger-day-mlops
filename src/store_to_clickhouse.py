import pandas as pd
import clickhouse_connect
from datetime import datetime

def store_to_clickhouse():
    df = pd.read_csv("data/processed/trigger_day_processed.csv")

    # Safety check
    print("CSV Columns:", list(df.columns))

    # Add ingestion time
    df["ingestion_time"] = datetime.now()

    client = clickhouse_connect.get_client(
        host="127.0.0.1",
        port=8123,
        username="default",
        password="admin123",   # 🔑 REQUIRED
        database="ivf_mlops"
    )



    columns = [
        "age",
        "amh_ng_ml",
        "cycle_day",
        "avg_follicle_size_mm",
        "follicle_count",
        "estradiol_pg_ml",
        "progesterone_ng_ml",
        "bmi",
        "basal_lh_miu_ml",
        "afc",
        "trigger_recommended",
        "ingestion_time"
    ]

    df = df[columns]   # schema lock 🔒

    client.insert(
        table="trigger_day_features",
        data=df.values.tolist(),
        column_names=columns
    )

    print(" Data stored in ClickHouse successfully")

if __name__ == "__main__":
    store_to_clickhouse()
