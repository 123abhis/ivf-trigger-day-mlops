from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "IVF_Trigger_Day_RF"

# This API resolves alias → version
try:
    mv = client.get_model_version_by_alias(
        name=model_name,
        alias="challenger"
    )
    print(f"Alias 'challenger' points to version: {mv.version}")
except Exception as e:
    print("Alias not found:", e)
