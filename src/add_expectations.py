import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

def add_expectations():
    context = gx.get_context(context_root_dir="gx")

    batch_request = RuntimeBatchRequest(
        datasource_name="ivf_filesystem_ds",
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="raw_trigger_day",
        runtime_parameters={
            "path": "data/raw/Trigger_Day_new_Dataset.csv"
        },
        batch_identifiers={
            "default_identifier_name": "raw_batch"
        },
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="trigger_day_expectations",
    )

    # ---------------- EXPECTATIONS ----------------

    validator.expect_column_to_exist("Patient_ID")
    validator.expect_column_values_to_not_be_null("Patient_ID")

    validator.expect_column_values_to_be_between(
        "Age", min_value=18, max_value=50
    )

    validator.expect_column_values_to_be_between(
        "AMH (ng/mL)", min_value=0.1, max_value=20
    )

    validator.expect_column_values_to_be_in_set(
        "Trigger_Recommended (0/1)", [0, 1]
    )

    validator.expect_column_values_to_be_between(
        "BMI", min_value=15, max_value=45
    )

    validator.save_expectation_suite()

    print("✅ Expectations added successfully")

if __name__ == "__main__":
    add_expectations()
