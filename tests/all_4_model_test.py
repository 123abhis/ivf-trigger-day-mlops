      
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
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# )

# from xgboost import XGBClassifier

#  --------------------------------------------------
#           ClickHouse wait
#  --------------------------------------------------

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
#     mlflow.set_tracking_uri("http://mlflow:5000")
#     mlflow.set_experiment("IVF_Trigger_Day_Models")

#     client = get_clickhouse_client()

#     df = client.query_df("""
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
#     """)

#     X = df.drop(columns=["trigger_recommended"])
#     y = df["trigger_recommended"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )

    
    
#     pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

#     models = {
#             "Logistic_Regression": Pipeline([
#                 ("scaler", StandardScaler()),
#                 ("model", LogisticRegression(
#                     max_iter=1000,
#                     class_weight="balanced"
#                 ))
#             ]),

#             "Random_Forest": RandomForestClassifier(
#                 n_estimators=300,
#                 random_state=42,
#                 class_weight="balanced",
#                 min_samples_leaf=5
#             ),
            
#             "Gradient_Boosting": GradientBoostingClassifier(
#             random_state=42,
#             ),
    

#             "XGBoost": XGBClassifier(
#                 n_estimators=300,
#                 max_depth=4,
#                 learning_rate=0.05,
#                 subsample=0.8,
#                 colsample_bytree=0.8,
#                 scale_pos_weight=pos_weight,
#                 eval_metric="logloss",
#                 random_state=42
#             )
#         }
    
    
    

#     for model_name, model in models.items():
#         with mlflow.start_run(run_name=model_name):
      
#             mlflow.set_tags({
#             "project": "ivf_trigger_day_mlops",
#             "author": "Abhishek Magadum",
#             "environment": "docker",
#             "model_name": model_name,
#             "algorithm": model.__class__.__name__
#         })

#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
            
#         # -------------- Logging the metrics -----------------

#             mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
            
#             mlflow.log_metric("f1_macro",f1_score(y_test, y_pred, average="macro"))
#             mlflow.log_metric("recall_positive",recall_score(y_test, y_pred, pos_label=1))
#             mlflow.log_metric("precision_positive", precision_score(y_test, y_pred, pos_label=1, zero_division=0))

#             mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))
            
            
#             #-------------------- Adding the parameters logging -----------------

#             if model_name == "Random_Forest":
#                 mlflow.log_param("n_estimators", model.n_estimators)
#                 mlflow.log_param("min_samples_leaf", model.min_samples_leaf)
#                 mlflow.log_param("class_weight", model.class_weight)
#                 mlflow.log_param("random_state", model.random_state)
                
#             if model_name == "logistic_regression":
#                 mlflow.log_param("class_weight", model.named_steps['model'].class_weight)
#                 mlflow.log_param("max_iter", model.named_steps['model'].max_iter)
                
#             if model_name == "xgboost":
#                 mlflow.log_param("n_estimators", model.n_estimators)
#                 mlflow.log_param("max_depth", model.max_depth)
#                 mlflow.log_param("learning_rate", model.learning_rate)
#                 mlflow.log_param("subsample", model.subsample)
#                 mlflow.log_param("colsample_bytree", model.colsample_bytree)
#                 mlflow.log_param("scale_pos_weight", model.scale_pos_weight)
#                 mlflow.log_param("eval_metric", model.eval_metric)

#             if model_name == "Gradient_Boosting":
                
#                 weights = y_train.map({0: 1, 1: 15})
#                 model.fit(X_train, y_train, sample_weight=weights)
#             else:
#                 model.fit(X_train, y_train)
                
                
#         #  -------------- setting the tags  -----------------
    
#             mlflow.set_tag("project", "ivf_trigger_day_mlops")
#             mlflow.set_tag("author", "Abhishek Magadum")
#             mlflow.set_tag("environment", "docker")

#             mlflow.sklearn.log_model(model, "model")
#             print(f"{model_name} trained successfully")
            
            

# if __name__ == "__main__":
#     train_model()
