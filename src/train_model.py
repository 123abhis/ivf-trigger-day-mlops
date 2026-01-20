import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble  import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# def train_model():
#     # Load processed data
#     df = pd.read_csv("data/raw/Trigger_Day_new_Dataset.csv")

#     # Features & target
#     X = df.drop(columns=["Trigger_Recommended (0/1)", "Patient_ID", "Visit_Date"])
#     y = df["Trigger_Recommended (0/1)"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     import mlflow
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
#         with mlflow.start_run(run_name=model_name, description="Baseline model for IVF trigger day prediction"):

        
#             mlflow.log_artifact("data/raw/Trigger_Day_new_Dataset.csv",artifact_path="dataset")

#         #  Dataset info
#             mlflow.set_tag("dataset_name", "Trigger_Day_new_Dataset.csv")
#             mlflow.set_tag("dataset_version", "v1.0")
#             mlflow.set_tag("data_source", "Hospital CSV export")
#             mlflow.set_tag("target_column", "Trigger_Recommended (0/1)")

#             # 🔹 TAGS (INSIDE run)
#             mlflow.set_tag("project", "ivf_trigger_day_mlops")
#             mlflow.set_tag("author", "Abhishek Magadum")
#             mlflow.set_tag("model_type", "RandomForest")
#             mlflow.set_tag("dataset", "Trigger_Day_new_Dataset")
#             mlflow.set_tag("stage", "baseline")
#             mlflow.set_tag("environment", "local")

#         model.fit(X_train, y_train)

#         y_pred = model.predict(X_test)

#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         roc_auc = roc_auc_score(y_test, y_pred)

#         # 🔹 Log params
#         mlflow.log_param("n_estimators", 100)
#         mlflow.log_param("max_depth", 5)

#         # 🔹 Log metrics
#         mlflow.log_metric("accuracy", acc)
#         mlflow.log_metric("f1_score", f1)
#         mlflow.log_metric("precision",precision )
#         mlflow.log_metric("recall", recall)
#         mlflow.log_metric("roc_auc", roc_auc )

#         # 🔹 Log model
#         mlflow.sklearn.log_model(model, "model")

#         print(f"Accuracy: {acc}")
#         print(f"F1 Score: {f1}")
#         print(f"precision: {precision}")
#         print(f"recall: {recall}")
#         print(f"roc_auc: {roc_auc}")

def train_model():
    import mlflow
    import mlflow.sklearn

    df = pd.read_csv("data/raw/Trigger_Day_new_Dataset.csv")

    X = df.drop(columns=["Trigger_Recommended (0/1)", "Patient_ID", "Visit_Date"])
    y = df["Trigger_Recommended (0/1)"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("IVF_Trigger_Day_Models")
    mlflow.end_run()  # safety reset

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

        with mlflow.start_run(run_name=model_name,
                              description="Baseline model for IVF trigger day prediction"):

            # 🔹 Train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 🔹 Metrics
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            # 🔹 Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("roc_auc", roc_auc)

            # 🔹 Log params (only if model supports them)
            if model_name == "Random_Forest":
                mlflow.log_param("n_estimators", model.n_estimators)

            # 🔹 Dataset artifact
            mlflow.log_artifact(
                "data/raw/Trigger_Day_new_Dataset.csv",
                artifact_path="dataset"
            )

            # 🔹 Tags
            mlflow.set_tag("project", "ivf_trigger_day_mlops")
            mlflow.set_tag("author", "Abhishek Magadum")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("stage", "baseline")
            mlflow.set_tag("environment", "local")

            # 🔹 Log model
            mlflow.sklearn.log_model(model, "model")

            print(f"\nModel: {model_name}")
            print(f"Accuracy: {acc}")
            print(f"F1 Score: {f1}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"ROC AUC: {roc_auc}")


if __name__ == "__main__":
    train_model()
