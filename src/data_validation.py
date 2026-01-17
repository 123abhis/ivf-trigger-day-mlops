import great_expectations as gx

def run_data_validation():
    context = gx.get_context()

    # 1. Get datasource
    datasource = context.get_datasource("ivf_datasource")

    # 2. Get dataframe asset (auto-inferred from folder)
    asset = datasource.get_asset("Trigger_Day_new_Dataset")

    # 3. Build batch request
    batch_request = asset.build_batch_request()

    # 4. Expectation suite name
    suite_name = "ivf_expectation_suite"

    # 5. Create or load expectation suite
    try:
        context.get_expectation_suite(suite_name)
        print("Expectation suite already exists. Loading it.")
    except Exception:
        print("Expectation suite not found. Creating a new one.")
        context.add_expectation_suite(expectation_suite_name=suite_name)

    # 6. Get validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # 7. Add core expectations (ML-focused)
    validator.expect_column_values_to_not_be_null("Trigger_Recommended (0/1)")
    validator.expect_column_values_to_be_in_set(
        "Trigger_Recommended (0/1)", [0, 1]
    )

    validator.expect_table_row_count_to_be_between(min_value=100)

    # 8. Save expectations
    validator.save_expectation_suite()
    print("Data validation completed successfully.")

if __name__ == "__main__":
    run_data_validation()














# import pandas as pd

# def validate_data():
#     df = pd.read_csv("data/raw/Trigger_Day_new_Dataset.csv")
#     print(df.head(5))
    
#     print(" Dataset shape: \n", df.shape)
    
#     print("Duplicate rows:\n")
#     print(df.duplicated().sum())

#     print("informative rows:\n")
#     print(df.info())

#     print("Statistical summary:\n")
#     print(df.describe())

#     print("Missing values:")
#     print(df.isnull().sum())
    
#     print("\nData types:")
#     print(df.dtypes)

#     assert df.shape[0] > 0, "Dataset is empty!"

#     print(" Data validation passed")

# if __name__ == "__main__":
#     validate_data()
