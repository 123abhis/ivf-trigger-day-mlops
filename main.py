from src.data_ingestion import ingest_data
from src.data_validation import  validate_raw_data
from src.data_preprocessing import preprocess_data


from src.store_to_clickhouse import store_to_clickhouse

def main():
    print("Starting IVF Trigger Day ML Pipeline")

    print("Data Ingestion started...")
    ingest_data()

    print("Data Validation started...")
    validate_raw_data()
    

    print("Data Preprocessing started...")
    preprocess_data()

    print(" Pipeline completed successfully!")


    print("Storing processed data to ClickHouse...")
    store_to_clickhouse()




if __name__ == "__main__":
    main()
