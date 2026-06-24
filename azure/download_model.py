"""
download_model.py
Downloads latest registered model from Azure ML Registry
into models/ folder for Docker build
"""
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os
import shutil

SUBSCRIPTION_ID = "1c2fd79b-ad21-4ad0-8d53-12de16650452"
RESOURCE_GROUP  = "loan-rg"
WORKSPACE_NAME  = "loan-ml-workspace"
MODEL_NAME      = "loan-classifier"
MODEL_VERSION   = "1"

ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

print(f">>> Downloading {MODEL_NAME} version {MODEL_VERSION} ...")

ml_client.models.download(
    name=MODEL_NAME,
    version=MODEL_VERSION,
    download_path="./models_temp",
)

# Copy model files to models/ folder
os.makedirs("models", exist_ok=True)
for f in os.listdir(f"./models_temp/{MODEL_NAME}/"):
    shutil.copy(
        f"./models_temp/{MODEL_NAME}/{f}",
        f"./models/{f}"
    )

print(">>> Model downloaded successfully to models/")
print(f">>> Files: {os.listdir('./models')}")