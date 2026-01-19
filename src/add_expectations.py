
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
        create_expectation_suite_with_name_if_missing=True
    )

    # ---- BASIC SCHEMA ----
    validator.expect_table_column_count_to_be_between(10, 20)
    validator.expect_column_to_exist("Patient_ID")

    # ---- BUSINESS RULES ----
    validator.expect_column_values_to_be_between("Age", 18, 50)
    validator.expect_column_values_to_be_between("BMI", 15, 45)
    validator.expect_column_values_to_be_between("AMH (ng/mL)", 0.1, 20)

    validator.expect_column_values_to_be_in_set(
        "Trigger_Recommended (0/1)", [0, 1]
    )

    # ---- NULL TOLERANCE (REALISTIC) ----
    validator.expect_column_values_to_not_be_null("Patient_ID")
    validator.expect_column_values_to_not_be_null("Age")

    validator.save_expectation_suite()
    print(" Expectations saved")

if __name__ == "__main__":
    add_expectations()
