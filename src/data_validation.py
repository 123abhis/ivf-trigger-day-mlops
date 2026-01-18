import great_expectations as gx
from great_expectations.core.batch import BatchRequest

def create_expectation_suite():
    context = gx.get_context()
    suite_name = "trigger_day_expectations"

    if suite_name in context.list_expectation_suite_names():
        print("Expectation suite already exists")
        return

    suite = context.add_expectation_suite(suite_name)

    batch_request = BatchRequest(
        datasource_name="ivf_filesystem_ds",
        data_connector_name="default_inferred_data_connector_name",
        data_asset_name="Trigger_Day_new_Dataset"
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite=suite
    )

    # Core validations
    validator.expect_table_row_count_to_be_between(100, 100000)
    validator.expect_column_values_to_not_be_null("Patient_ID")
    validator.expect_column_values_to_be_in_set("Trigger_Recommended", [0, 1])

    # Clinical sanity checks
    validator.expect_column_values_to_be_between("Age", 18, 55)
    validator.expect_column_values_to_be_between("AMH", 0.1, 20)
    validator.expect_column_values_to_be_between("Avg_Follicle_Size_mm", 5, 30)

    validator.save_expectation_suite()
    print("Expectation suite created successfully")

if __name__ == "__main__":
    create_expectation_suite()
