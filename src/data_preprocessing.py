import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data():
    # 1. Load dataset
    df = pd.read_csv("data/raw/Trigger_Day_new_Dataset.csv")

    # 2. Separate features and target
    X = df.drop("Trigger_Recommended (0/1)", axis=1)
    y = df["Trigger_Recommended (0/1)"]

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
