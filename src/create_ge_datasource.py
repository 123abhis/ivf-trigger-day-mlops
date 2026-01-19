import great_expectations as gx

def create_datasource():
    context = gx.get_context(context_root_dir="gx")

    datasource_name = "ivf_filesystem_ds"

    if datasource_name in context.list_datasources():
        print("ℹ Datasource already exists")
        return

    context.add_datasource(
        name=datasource_name,
        class_name="Datasource",
        execution_engine={
            "class_name": "PandasExecutionEngine"
        },
        data_connectors={
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"]
            }
        }
    )

    print(" Datasource created successfully")

if __name__ == "__main__":
    create_datasource()
