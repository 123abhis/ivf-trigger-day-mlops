import great_expectations as gx

def add_asset():
    context = gx.get_context()
    datasource = context.sources.get("ivf_filesystem_ds")

    asset_name = "trigger_day_csv"

    if asset_name in datasource.assets:
        print("Asset already exists")
        return

    datasource.add_csv_asset(
        name=asset_name,
        batching_regex=r"Trigger_Day_new_Dataset\.csv"
    )
    print("Asset added")

if __name__ == "__main__":
    add_asset()
