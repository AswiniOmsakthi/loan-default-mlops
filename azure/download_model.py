"""
download_model.py
Downloads latest registered model from Azure ML Registry and version also happens
into models/ folder for Docker build
Written by: MLOps Engineer
"""
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os
import shutil

SUBSCRIPTION_ID = "1c2fd79b-ad21-4ad0-8d53-12de16650452"
RESOURCE_GROUP  = "loan-rg"
WORKSPACE_NAME  = "loan-ml-workspace"
MODEL_NAME      = "loan-classifier"

# ── Connect to Azure ML ──────────────────────────────────────
print(">>> Connecting to Azure ML ...")
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# ── Get latest version ────────────────────────────────────────
print(f">>> Getting latest version of {MODEL_NAME} ...")
models = list(ml_client.models.list(name=MODEL_NAME))
latest_version = str(max([int(m.version) for m in models]))
print(f">>> Latest version: {latest_version}")

# ── Download model ────────────────────────────────────────────
print(f">>> Downloading {MODEL_NAME} version {latest_version} ...")
ml_client.models.download(
    name=MODEL_NAME,
    version=latest_version,
    download_path="./models_temp",
)

# ── Copy to models/ folder ────────────────────────────────────
os.makedirs("models", exist_ok=True)
for root, dirs, files in os.walk("./models_temp"):
    for file in files:
        if file.endswith(".joblib") or file.endswith(".json"):
            shutil.copy(
                os.path.join(root, file),
                f"./models/{file}"
            )
            print(f">>> Copied: {file}")

print(f">>> Files in models/: {os.listdir('./models')}")
print(">>> Download complete")
