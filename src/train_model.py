
# ---------------------------------  model registration added version ----------------------------- #

import time
import pandas as pd
import mlflow
import mlflow.sklearn

from clickhouse_connect import get_client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from mlflow.models.signature import infer_signature


# --------------------------------------------------
# ClickHouse wait
# --------------------------------------------------
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


def train_model():

    # ---------------- MLflow config ----------------
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("IVF_Trigger_Day_Models")

    # ---------------- Load data ----------------
    client = get_clickhouse_client()

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
            cluster_id,
            trigger_recommended
        FROM trigger_day_features
    """)

    X = df.drop(columns=["trigger_recommended"])
    y = df["trigger_recommended"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------------- Model ----------------
    
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=5
    )

    # ---------------- MLflow Run ----------------
    with mlflow.start_run(run_name="rf_trigger_day_v1"):

        mlflow.set_tags({
            "project": "ivf_trigger_day_mlops",
            "author": "Abhishek Magadum",
            "environment": "docker",
            "model_name": "Random_Forest",
            "algorithm": "RandomForestClassifier"
        })

        # Train
        rf.fit(X_train, y_train)

        # Predict
        y_pred = rf.predict(X_test)

        # Metrics
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("recall_positive", recall_score(y_test, y_pred, pos_label=1))
        mlflow.log_metric("precision_positive", precision_score(y_test, y_pred, pos_label=1))
        mlflow.log_metric("roc_auc", roc_auc)

        # Params
        mlflow.log_param("n_estimators", rf.n_estimators)
        mlflow.log_param("min_samples_leaf", rf.min_samples_leaf)
        mlflow.log_param("class_weight", rf.class_weight)
        mlflow.log_param("random_state", rf.random_state)

        # Signature
        signature = infer_signature(X_test, y_pred)

        # Log & Register Model
        mlflow.sklearn.log_model(
            sk_model=rf,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[:5],
            registered_model_name="IVF_Trigger_Day_RF"
        )

        print("Random Forest trained and registered successfully")


if __name__ == "__main__":
    train_model()
