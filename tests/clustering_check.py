# import pandas as pd

# df = pd.read_csv("data/processed/trigger_day_processed.csv")

# # 1️⃣ Check cluster distribution
# print(df["cluster_id"].value_counts())

# # 2️⃣ Same patient should have SAME cluster_id
# patient_cluster_check = (
#     df.groupby("patient_id")["cluster_id"]
#       .nunique()
# )

# print("Patients with >1 cluster:",
#       (patient_cluster_check > 1).sum())

# # 3️⃣ Check relation with target
# print(pd.crosstab(df["cluster_id"], df["trigger_recommended"], normalize="index"))


# import pandas as pd

# df = pd.read_csv("data/processed/trigger_day_processed.csv")

# print(
#     pd.crosstab(
#         df["cluster_id"],
#         df["trigger_recommended"],
#         normalize="index"
#     )
# )


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.metrics import (
#     accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# )


df = pd.read_csv("data/processed/trigger_day_processed.csv")

X = df[["cluster_id"]]   # ONLY cluster
y = df["trigger_recommended"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("----- Using Random Forest Classifier -----")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("----- Using Random Forest Classifier complete successfully -----")


print("----- Using Gradient Boosting Classifier -----")
model1 = GradientBoostingClassifier(random_state=42)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
print(classification_report(y_test, y_pred1))
print("----- Using Gradient Boosting Classifier complete successfully -----")


print(" ---------using the logistic regression model ----------")
from sklearn.linear_model import LogisticRegression
model2 = LogisticRegression(max_iter=1000)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print(classification_report(y_test, y_pred2))
print(" ---------using the logistic regression model  complete successfully ----------")

print("Prediction distribution:")
print(pd.Series(y_pred).value_counts())

print("Actual distribution:")
print(y_test.value_counts())





