# import pandas as pd
# import mlflow
# import mlflow.sklearn   

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble  import GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# def train_model():
#     import mlflow
#     import mlflow.sklearn

#     df = pd.read_csv("data/processed/trigger_day_processed.csv")

#     X = df.drop(columns=["trigger_recommended"])
#     y = df["trigger_recommended"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     mlflow.set_experiment("IVF_Trigger_Day_Models")
#     mlflow.end_run()  # safety reset

#     models = {
#         "Logistic_Regression": Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", LogisticRegression(max_iter=1000))
#         ]),
#         "Random_Forest": RandomForestClassifier(
#             n_estimators=200,
#             random_state=42
#         ),
#         "Gradient_Boosting": GradientBoostingClassifier(
#             random_state=42
#         )
#     }

#     for model_name, model in models.items():

#         with mlflow.start_run(run_name=model_name,
#                               description="Baseline model for IVF trigger day prediction"):

#             # 🔹 Train
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             # 🔹 Metrics
#             acc = accuracy_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
#             recall = recall_score(y_test, y_pred)
#             roc_auc = roc_auc_score(y_test, y_pred)

#             # 🔹 Log metrics
#             mlflow.log_metric("accuracy", acc)
#             mlflow.log_metric("f1_score", f1)
#             mlflow.log_metric("precision", precision)
#             mlflow.log_metric("recall", recall)
#             mlflow.log_metric("roc_auc", roc_auc)

#             # 🔹 Log params (only if model supports them)
#             if model_name == "Random_Forest":
#                 mlflow.log_param("n_estimators", model.n_estimators)

#             # 🔹 Dataset artifact
#             mlflow.log_artifact(
#                 "data/processed/trigger_day_processed.csv",
#                 artifact_path="dataset"
#             )

#             # 🔹 Tags
#             mlflow.set_tag("project", "ivf_trigger_day_mlops")
#             mlflow.set_tag("author", "Abhishek Magadum")
#             mlflow.set_tag("model_name", model_name)
#             mlflow.set_tag("stage", "baseline")
#             mlflow.set_tag("environment", "local")

#             # 🔹 Log model
#             mlflow.sklearn.log_model(model, "model")

#             print(f"\nModel: {model_name}")
#             print(f"Accuracy: {acc}")
#             print(f"F1 Score: {f1}")
#             print(f"Precision: {precision}")
#             print(f"Recall: {recall}")
#             print(f"ROC AUC: {roc_auc}")


# if __name__ == "__main__":
#     train_model()







# import time
# import pandas as pd
# import mlflow
# import mlflow.sklearn

# from clickhouse_connect import get_client

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


# # --------------------------------------------------
# # Wait for ClickHouse to be ready
# # --------------------------------------------------
# def get_clickhouse_client():
#     for i in range(10):
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
#         except Exception as e:
#             print(f"Waiting for ClickHouse... ({i+1}/10)")
#             time.sleep(5)

#     raise RuntimeError("ClickHouse not available")


# def train_model():
#     client = get_clickhouse_client()

#     query = """
#         SELECT
#             age,
#             amh_ng_ml,
#             cycle_day,
#             avg_follicle_size_mm,
#             follicle_count,
#             estradiol_pg_ml,
#             progesterone_ng_ml,
#             bmi,
#             basal_lh_miu_ml,
#             afc,
#             cluster_id,
#             trigger_recommended
#         FROM trigger_day_features
#     """

#     data = client.query(query).result_rows

#     columns = [
#         "age",
#         "amh_ng_ml",
#         "cycle_day",
#         "avg_follicle_size_mm",
#         "follicle_count",
#         "estradiol_pg_ml",
#         "progesterone_ng_ml",
#         "bmi",
#         "basal_lh_miu_ml",
#         "afc",
#         "cluster_id",
#         "trigger_recommended"
#     ]

#     df = pd.DataFrame(data, columns=columns)

#     X = df.drop(columns=["trigger_recommended"])
#     y = df["trigger_recommended"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

#     mlflow.set_experiment("IVF_Trigger_Day_Models")
#     mlflow.end_run()

#     models = {
#         "Logistic_Regression": Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", LogisticRegression(max_iter=1000))
#         ]),
#         "Random_Forest": RandomForestClassifier(
#             n_estimators=200,
#             random_state=42
#         ),
#         "Gradient_Boosting": GradientBoostingClassifier(
#             random_state=42
#         )
#     }

#     for model_name, model in models.items():
#         with mlflow.start_run(run_name=model_name):

#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
#             mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
#             mlflow.log_metric("precision", precision_score(y_test, y_pred))
#             mlflow.log_metric("recall", recall_score(y_test, y_pred))
#             mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))

#             if model_name == "Random_Forest":
#                 mlflow.log_param("n_estimators", model.n_estimators)

           

#             mlflow.set_tag("project", "ivf_trigger_day_mlops")
#             mlflow.set_tag("author", "Abhishek Magadum")
#             mlflow.set_tag("environment", "docker")

#             mlflow.sklearn.log_model(model, "model")

#             print(f"{model_name} trained successfully")


# if __name__ == "__main__":
#     train_model()









# import time
# import pandas as pd
# import mlflow
# import mlflow.sklearn


# from mlflow.data import from_pandas


# from clickhouse_connect import get_client

# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# )

# # --------------------------------------------------
# # ClickHouse connection wait
# # --------------------------------------------------
# def get_clickhouse_client():
#     for i in range(10):
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
#             print(f"Waiting for ClickHouse... ({i+1}/10)")
#             time.sleep(5)

#     raise RuntimeError("ClickHouse not available")


# def train_model():
#     # 🔑 REQUIRED inside Docker
#     mlflow.set_tracking_uri("http://mlflow:5000")
#     mlflow.set_experiment("IVF_Trigger_Day_Models")

#     client = get_clickhouse_client()

#     query = """
#         SELECT
#             age,
#             amh_ng_ml,
#             cycle_day,
#             avg_follicle_size_mm,
#             follicle_count,
#             estradiol_pg_ml,
#             progesterone_ng_ml,
#             bmi,
#             basal_lh_miu_ml,
#             afc,
#             cluster_id,
#             trigger_recommended
#         FROM trigger_day_features
#     """

#     df = client.query_df(query)

#     X = df.drop(columns=["trigger_recommended"])
#     y = df["trigger_recommended"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )


#     models = {
#         "Logistic_Regression": Pipeline([
#             ("scaler", StandardScaler()),
#             ("model", LogisticRegression(max_iter=1000))
#         ]),
#         "Random_Forest": RandomForestClassifier(
#             n_estimators=200,
#             random_state=42
#         ),
#         "Gradient_Boosting": GradientBoostingClassifier(
#             random_state=42
#         )
#     }

#     for model_name, model in models.items():
#         with mlflow.start_run(run_name=model_name):

#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             # Metrics
#             mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
#             mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
#             mlflow.log_metric("precision", precision_score(y_test, y_pred))
#             mlflow.log_metric("recall", recall_score(y_test, y_pred))
#             mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))

#             if model_name == "Random_Forest":
#                 mlflow.log_param("n_estimators", model.n_estimators)

#             # ✅ DATASET LOGGING — SAFE VERSION
#             # dataset = from_pandas(df)
#             # mlflow.log_input(dataset, context="training")

#             # Tags
#             mlflow.set_tag("project", "ivf_trigger_day_mlops")
#             mlflow.set_tag("author", "Abhishek Magadum")
#             mlflow.set_tag("environment", "docker")
#             mlflow.set_tag("model_name", model_name)

#             # Model
#             mlflow.sklearn.log_model(model, "model")

#             print(f"{model_name} trained successfully")

# if __name__ == "__main__":
#     train_model()





        # with mlflow.start_run(run_name=model_name):

        #     # 🔗 attach dataset to run
        #     mlflow.log_input(dataset, context="training")

        #     model.fit(X_train, y_train)
        #     y_pred = model.predict(X_test)

        #     # Metrics
        #     mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        #     mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
        #     mlflow.log_metric("precision", precision_score(y_test, y_pred))
        #     mlflow.log_metric("recall", recall_score(y_test, y_pred))
        #     mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))

        #     if model_name == "Random_Forest":
        #         mlflow.log_param("n_estimators", model.n_estimators)

        #     # Tags
        #     mlflow.set_tag("project", "ivf_trigger_day_mlops")
        #     mlflow.set_tag("author", "Abhishek Magadum")
        #     mlflow.set_tag("environment", "docker")

        #     # ✅ Register model
        #     mlflow.sklearn.log_model(
        #         model,
        #         artifact_path="model",
        #         registered_model_name=model_name
        #     )

        #     print(f"{model_name} trained and logged to MLflow")
        
        
        
        
        
        
        
        
import time
import pandas as pd
import mlflow
import mlflow.sklearn

from clickhouse_connect import get_client

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

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
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("IVF_Trigger_Day_Models")

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
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "Random_Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42
        ),
        "Gradient_Boosting": GradientBoostingClassifier(
            random_state=42
        )
    }

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            
            mlflow.set_tags({
            "project": "ivf_trigger_day_mlops",
            "author": "Abhishek Magadum",
            "environment": "docker",
            "model_name": model_name,
            "algorithm": model.__class__.__name__
        })

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))

            if model_name == "Random_Forest":
                mlflow.log_param("n_estimators", model.n_estimators)

            mlflow.set_tag("project", "ivf_trigger_day_mlops")
            mlflow.set_tag("author", "Abhishek Magadum")
            mlflow.set_tag("environment", "docker")

            mlflow.sklearn.log_model(model, "model")

            print(f"{model_name} trained successfully")


if __name__ == "__main__":
    train_model()
