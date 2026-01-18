import great_expectations as gx
from pathlib import Path

def create_datasource():
    context = gx.get_context()

    datasource_name = "ivf_filesystem_ds"

    if datasource_name in [ds["name"] for ds in context.list_datasources()]:
        print("Datasource already exists")
        return

    context.add_datasource(
        name=datasource_name,
        class_name="Datasource",
        execution_engine={
            "class_name": "PandasExecutionEngine"
        },
        data_connectors={
            "default_inferred_data_connector_name": {
                "class_name": "InferredAssetFilesystemDataConnector",
                "base_directory": str(Path("data/raw").resolve()),
                "default_regex": {
                    "group_names": ["data_asset_name"],
                    "pattern": r"(.*)\.csv"
                }
            }
        }
    )

    print("Datasource created successfully")

if __name__ == "__main__":
    create_datasource()

