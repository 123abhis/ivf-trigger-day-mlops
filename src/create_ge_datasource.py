import great_expectations as gx

context = gx.get_context()

datasource = context.sources.add_pandas_filesystem(
    name="ivf_datasource",
    base_directory="data/raw"
)

print("Datasource created:", datasource.name)

print("Assets found:")
for asset in datasource.assets:
    print("-", asset.name)
