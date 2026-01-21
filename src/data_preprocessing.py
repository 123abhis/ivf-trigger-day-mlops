import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data():
    df = pd.read_csv("data/raw/Trigger_Day_new_Dataset.csv")

    df = df.rename(columns={
        "Patient_ID": "patient_id",
        "Age": "age",
        "AMH (ng/mL)": "amh_ng_ml",
        "Day": "cycle_day",
        "Avg_Follicle_Size_mm": "avg_follicle_size_mm",
        "Follicle_Count": "follicle_count",
        "Estradiol_pg_mL": "estradiol_pg_ml",
        "Progesterone_ng_mL": "progesterone_ng_ml",
        "BMI": "bmi",
        "Basal_LH_mIU_mL": "basal_lh_miu_ml",
        "AFC": "afc",
        "Visit_Date": "visit_date",
        "Trigger_Recommended (0/1)": "trigger_recommended"
    })

    # Drop non-ML columns
    df = df.drop(columns=["patient_id", "visit_date"])

    # Save processed data
    df.to_csv("data/processed/trigger_day_processed.csv", index=False)

    print(" Data preprocessing completed")
    
    
    X = df.drop("trigger_recommended", axis=1)
    y = df["trigger_recommended"]

    # 3. Identify column types
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    print("Numerical columns:", list(num_cols))
    print("Categorical columns:", list(cat_cols))

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # 6. Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 7. Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # 8. Fit on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    print("Preprocessing completed successfully")
    print("Train shape:", X_train_processed.shape)
    print("Test shape:", X_test_processed.shape)

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    preprocess_data()
