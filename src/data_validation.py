
import great_expectations as gx
from great_expectations.core.batch import RuntimeBatchRequest

def validate_raw_data():
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

    checkpoint_name = "raw_trigger_day_checkpoint"

    # Add checkpoint
    context.add_or_update_checkpoint(
        name=checkpoint_name,
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": "trigger_day_expectations",
            }
        ],
    )

    # Run checkpoint (THIS updates index.html stats)
    results = context.run_checkpoint(checkpoint_name=checkpoint_name)

    print("Validation success:", results["success"])

    context.build_data_docs()
    print("Data Docs rebuilt successfully")

    return results["success"]


if __name__ == "__main__":
    validate_raw_data()
