from src.data_ingestion import ingest_data
from src.data_validation import validate_data
from src.data_preprocessing import preprocess_data




def main():
    print("Starting IVF Trigger Day ML Pipeline")

    print("Data Ingestion started...")
    ingest_data()

    

    print("Data Preprocessing started...")
    preprocess_data()

    print(" Pipeline completed successfully!")


if __name__ == "__main__":
    main()
