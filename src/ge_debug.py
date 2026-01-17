import great_expectations as gx

context = gx.get_context()
datasource = context.get_datasource("ivf_datasource")

print("Datasource:", datasource.name)

print("\nAvailable data assets:")
for asset in datasource.assets:
    print("-", asset.name)
