# from src.data_ingestion import ingest_data
# from src.data_validation import  validate_raw_data
# from src.data_preprocessing import preprocess_data


# from src.store_to_clickhouse import store_to_clickhouse

# def main():
#     print("Starting IVF Trigger Day ML Pipeline")

#     print("Data Ingestion started...")
#     ingest_data()

#     print("Data Validation started...")
#     validate_raw_data()
    

#     print("Data Preprocessing started...")
#     preprocess_data()

#     print(" Pipeline completed successfully!")


#     print("Storing processed data to ClickHouse...")
#     store_to_clickhouse()




# if __name__ == "__main__":
#     main()




import time
import os
from src.clickhouse_connector import ClickHouseConnector
from src.data_ingestion import ingest_data
from src.data_preprocessing import preprocess_data
from src.train_model import train_and_evaluate_models
import pandas as pd

def main():
    print("=" * 60)
    print("🚀 Starting MLOps Pipeline")
    print("=" * 60)
    
    # Wait for services to be ready
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Step 1: Connect to ClickHouse
    print("\n📊 Step 1: Connecting to ClickHouse...")
    ch_connector = ClickHouseConnector()
    
    max_retries = 5
    for attempt in range(max_retries):
        if ch_connector.connect():
            break
        print(f"Retry {attempt + 1}/{max_retries}...")
        time.sleep(5)
    else:
        print("❌ Failed to connect to ClickHouse after multiple attempts")
        return
    
    # Create tables
    ch_connector.create_tables()
    
    # Step 2: Data Ingestion
    print("\n📥 Step 2: Running data ingestion...")
    raw_data = ingest_data()
    
    # Step 3: Data Preprocessing
    print("\n🔧 Step 3: Running data preprocessing...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(raw_data)
    
    # Step 4: Store processed data in ClickHouse
    print("\n💾 Step 4: Storing processed data in ClickHouse...")
    processed_df = pd.concat([X_train, X_test], axis=0)
    ch_connector.insert_processed_data(raw_data)  # Store the full dataset with all columns
    
    # Step 5: Train models
    print("\n🤖 Step 5: Training models...")
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Log training results to ClickHouse
    for model_name, metrics in results.items():
        ch_connector.log_training_run(
            run_id=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            model_name=model_name,
            metrics=metrics
        )
    
    print("\n" + "=" * 60)
    print("✅ Pipeline completed successfully!")
    print("=" * 60)
    print(f"\n📈 MLflow UI: http://localhost:5000")
    print(f"🗄️  ClickHouse: http://localhost:8123")
    
    # Keep MLflow UI running
    print("\n⏳ Keeping services alive... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()