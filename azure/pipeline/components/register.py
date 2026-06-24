"""
register.py
Pipeline Step 4 — Register model to Azure ML
Written by: MLOps Engineer
"""
import argparse
import json
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model",   type=str)
    parser.add_argument("--input_metrics", type=str)
    parser.add_argument("--min_roc_auc",   type=float, default=0.65)
    args = parser.parse_args()

    # ── Read metrics ─────────────────────────────────────────────
    with open(os.path.join(args.input_metrics, "metrics.json")) as f:
        metrics = json.load(f)

    roc_auc  = metrics["roc_auc"]
    accuracy = metrics["accuracy"]
    f1       = metrics["f1_score"]

    print(f">>> ROC-AUC  : {roc_auc}")
    print(f">>> Accuracy : {accuracy}")
    print(f">>> F1 Score : {f1}")
    print(f">>> Threshold: {args.min_roc_auc}")

    # ── Check threshold ──────────────────────────────────────────
    if roc_auc < args.min_roc_auc:
        print(">>> FAILED — ROC-AUC below threshold")
        print(">>> Model will NOT be registered")
        return

    # ── Connect using AzureMLOnBehalfOfCredential ────────────────
    print(">>> Connecting to Azure ML ...")
    credential = AzureMLOnBehalfOfCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=os.environ["AZUREML_ARM_SUBSCRIPTION"],
        resource_group_name=os.environ["AZUREML_ARM_RESOURCEGROUP"],
        workspace_name=os.environ["AZUREML_ARM_WORKSPACE_NAME"],
    )
    print(">>> Connected successfully")

    # ── Register model ───────────────────────────────────────────
    print(">>> Registering model ...")
    model = Model(
        path=os.path.join(args.input_model, "model.joblib"),
        name="loan-classifier",
        description=f"Loan GradientBoosting — ROC-AUC: {roc_auc}",
        type=AssetTypes.CUSTOM_MODEL,
        tags={
            "roc_auc":   str(roc_auc),
            "accuracy":  str(accuracy),
            "f1_score":  str(f1),
            "algorithm": "GradientBoostingClassifier",
            "framework": "scikit-learn",
            "stage":     "production",
        },
    )

    registered = ml_client.models.create_or_update(model)
    print(f">>> Registered successfully!")
    print(f"    Name    : {registered.name}")
    print(f"    Version : {registered.version}")
    print(f"    ROC-AUC : {roc_auc}")

if __name__ == "__main__":
    main()