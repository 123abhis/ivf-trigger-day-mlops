import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote

def ingest_data():
    # Load raw data
    df = pd.read_csv("data/raw/Trigger_Day_new_Dataset.csv")
    print(df.head(5))
    
    # Normalize Patient_ID
    df["Patient_ID"] = df["Patient_ID"].str.upper()

    # Fill missing numeric values with median
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    df.to_csv("data/raw/Trigger_Day_new_Dataset.csv", index=False)

    print(" Data ingestion & normalization completed")

    # Database config
    user = "root"
    password = "abhi123"
    db = "ivf_project_db"

    engine = create_engine(
        f"mysql+pymysql://{user}:{quote(password)}@localhost/{db}"
    )

    # Store in DB
    df.to_sql(
        "trigger_day_data",
        con=engine,
        if_exists="replace",
        index=False
    )
    
    dff = pd.read_sql("select* from trigger_day_data", con=engine)
    print(dff.head(5))

    print("Data ingestion completed")
    
    print(dff.columns)
    print(f"Data shape: {dff.shape}")
    print(f"Data types: {dff.dtypes}")
    print(f"Data summary: {dff.describe()}")
    print(f"Data head: {dff.head(5)}")
    print(f"Data null values:\n{dff.isnull().sum()}")
    print(f"Data unique values:\n{dff.nunique()}")

if __name__ == "__main__":
    ingest_data()
