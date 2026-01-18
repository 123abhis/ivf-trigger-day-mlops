import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

def validate_raw_data():
    context = gx.get_context(context_root_dir="gx")

    suite_name = "raw_trigger_day_suite"
    datasource_name = "ivf_filesystem_ds"

    if suite_name not in [s.expectation_suite_name for s in context.list_expectation_suites()]:
        context.create_expectation_suite(suite_name)
        print(f"✅ Created suite: {suite_name}")

    batch_request = RuntimeBatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="raw_trigger_day",
        runtime_parameters={
            "path": "data/raw/Trigger_Day_new_Dataset.csv"
        },
        batch_identifiers={
            "default_identifier_name": "raw_batch"
        }
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # ---------- Expectations ----------
    validator.expect_column_values_to_not_be_null("Patient_ID")
    validator.expect_column_values_to_be_between("Age", 18, 50)
    validator.expect_column_values_to_be_between("BMI", 15, 40)
    validator.expect_column_values_to_be_between("Day", 1, 20)
    validator.expect_column_values_to_match_regex("Patient_ID", r"^[Pp]\d{4}$")
    validator.expect_column_values_to_be_between("Trigger_Recommended (0/1)", 0, 1)

    result = validator.validate()

    validator.save_expectation_suite()
    context.build_data_docs()

    print("📊 Validation success:", result["success"])
    print("🌐 Data Docs generated")

if __name__ == "__main__":
    validate_raw_data()
