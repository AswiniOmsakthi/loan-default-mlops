"""
register_model.py
-----------------
Registers model.joblib into Azure ML Model Registry.
Run: python azure/register_model.py
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
import json
import os

# ── Azure ML config ────────────────────────────────────────────
SUBSCRIPTION_ID  = "1c2fd79b-ad21-4ad0-8d53-12de16650452"
RESOURCE_GROUP   = "loan-rg"
WORKSPACE_NAME   = "loan-ml-workspace"

# ── Paths ───────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(ROOT_DIR, "models", "model.joblib")
METRICS_PATH = os.path.join(ROOT_DIR, "models", "metrics.json")

# ── Load metrics ────────────────────────────────────────────────
with open(METRICS_PATH) as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]
print(f">>> Model accuracy: {accuracy}")

roc_auc = metrics["roc_auc"]
print(f">>> Model ROC AUC: {roc_auc}")

f1_score = metrics["f1_score"]
print(f">>> Model F1 Score: {f1_score}")

# ── Connect to Azure ML ─────────────────────────────────────────
print(">>> Connecting to Azure ML Workspace ...")
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# ── Register model ──────────────────────────────────────────────
print(">>> Registering model ...")
model = Model(
    path=MODEL_PATH,
    name="loan-classifier",
    description=f"Loan Gradient Boosting Classifier — accuracy: {accuracy}",
    type=AssetTypes.CUSTOM_MODEL,
    tags={
        "accuracy":    str(accuracy),
        "roc_auc":     str(roc_auc),
        "f1_score":    str(f1_score),
        "algorithm":   "GradientBoostingClassifier",
        "framework":   "scikit-learn",
        "stage":       "production",
    },
)

registered_model = ml_client.models.create_or_update(model)

print(f">>> Model registered successfully!")
print(f"    Name    : {registered_model.name}")
print(f"    Version : {registered_model.version}")
print(f"    Accuracy: {accuracy}")
print(f"    ROC AUC: {roc_auc}")
print(f"    F1 Score: {f1_score}")