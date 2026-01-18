import great_expectations as gx
from great_expectations.core.batch import BatchRequest

def add_raw_asset():
    context = gx.get_context()

    datasource_name = "ivf_filesystem_ds"

    batch_request = BatchRequest(
        datasource_name=datasource_name,
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="raw_trigger_day",
        runtime_parameters= {
            "path": "data/raw/Trigger_Day_new_Dataset.csv"
        },
        batch_identifiers= {
            "default_identifier_name": "raw_batch"
        },
    )

    context.get_validator(
        batch_request=batch_request,
        expectation_suite_name="raw_trigger_day_suite"
    )

    print(" Raw data asset registered")

if __name__ == "__main__":
    add_raw_asset()
