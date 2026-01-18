import great_expectations as gx

def run_checkpoint():
    context = gx.get_context()

    checkpoint = context.add_or_update_checkpoint(
        name="trigger_day_checkpoint",
        validations=[
            {
                "batch_request": {
                    "datasource_name": "ivf_filesystem_ds",
                    "data_connector_name": "default_inferred_data_connector_name",
                    "data_asset_name": "Trigger_Day_new_Dataset"
                },
                "expectation_suite_name": "trigger_day_expectations"
            }
        ]
    )

    result = checkpoint.run()
    print("Validation success:", result["success"])

if __name__ == "__main__":
    run_checkpoint()
