import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote

def ingest_data():
    # Load raw data
    df = pd.read_csv("data/raw/Trigger_Day_new_Dataset.csv")
    print(df.head(5))

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

if __name__ == "__main__":
    ingest_data()
