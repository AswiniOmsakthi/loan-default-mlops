"""
pipeline.py
Main pipeline — connects all 4 steps and submits to Azure ML
Written by: MLOps Engineer
Run: python azure/pipeline/pipeline.py
"""
import os
from azure.ai.ml import MLClient, Input, Output
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import CommandComponent, Environment
from azure.identity import DefaultAzureCredential

# ── Your Azure details ───────────────────────────────────────────
SUBSCRIPTION_ID = "1c2fd79b-ad21-4ad0-8d53-12de16650452"
RESOURCE_GROUP  = "loan-rg"
WORKSPACE_NAME  = "loan-ml-workspace"

# ── Connect to Azure ML ──────────────────────────────────────────
print(">>> Connecting to Azure ML ...")
ml_client = MLClient(
    DefaultAzureCredential(),
    SUBSCRIPTION_ID,
    RESOURCE_GROUP,
    WORKSPACE_NAME,
)

# ── Environment — what packages each step needs ──────────────────
env = Environment(
    name="loan-pipeline-env",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file={
        "name": "loan-env",
        "channels": ["defaults"],
        "dependencies": [
            "python=3.10",
            "pip",
            {"pip": [
                "scikit-learn==1.5.0",
                "pandas==2.2.2",
                "numpy==1.26.4",
                "joblib==1.4.2",
                "azure-ai-ml",
                "azure-identity",
            ]},
        ],
    },
)

# ── Step 1 — Data Prep ───────────────────────────────────────────
data_prep_component = CommandComponent(
    name="data_prep",
    display_name="Step 1 — Data Preparation",
    code="./components",
    command="python data_prep.py --output_data ${{outputs.output_data}}",
    environment=env,
    outputs={
        "output_data": Output(type="uri_folder"),
    },
)

# ── Step 2 — Train ───────────────────────────────────────────────
train_component = CommandComponent(
    name="train_model",
    display_name="Step 2 — Train Model",
    code="./components",
    command=(
        "python train.py"
        " --input_data ${{inputs.input_data}}"
        " --output_model ${{outputs.output_model}}"
    ),
    environment=env,
    inputs={
        "input_data": Input(type="uri_folder"),
    },
    outputs={
        "output_model": Output(type="uri_folder"),
    },
)

# ── Step 3 — Evaluate ────────────────────────────────────────────
evaluate_component = CommandComponent(
    name="evaluate_model",
    display_name="Step 3 — Evaluate Model",
    code="./components",
    command=(
        "python evaluate.py"
        " --input_data ${{inputs.input_data}}"
        " --input_model ${{inputs.input_model}}"
        " --output_metrics ${{outputs.output_metrics}}"
    ),
    environment=env,
    inputs={
        "input_data":  Input(type="uri_folder"),
        "input_model": Input(type="uri_folder"),
    },
    outputs={
        "output_metrics": Output(type="uri_folder"),
    },
)

# ── Step 4 — Register ────────────────────────────────────────────
register_component = CommandComponent(
    name="register_model",
    display_name="Step 4 — Register Model",
    code="./components",
    command=(
        "python register.py"
        " --input_model ${{inputs.input_model}}"
        " --input_metrics ${{inputs.input_metrics}}"
        " --min_roc_auc 0.65"
    ),
    environment=env,
    inputs={
        "input_model":   Input(type="uri_folder"),
        "input_metrics": Input(type="uri_folder"),
    },
)

# ── Connect all 4 steps into pipeline ───────────────────────────
@pipeline(
    name="loan-retraining-pipeline",
    description="Loan default classification — data_prep → train → evaluate → register",
    default_compute="serverless",
)
def loan_pipeline():
    step1 = data_prep_component()

    step2 = train_component(
        input_data=step1.outputs.output_data,
    )

    step3 = evaluate_component(
        input_data=step1.outputs.output_data,
        input_model=step2.outputs.output_model,
    )

    step4 = register_component(
        input_model=step2.outputs.output_model,
        input_metrics=step3.outputs.output_metrics,
    )

# ── Submit to Azure ML ───────────────────────────────────────────
if __name__ == "__main__":
    print(">>> Building pipeline ...")
    pipeline_job = loan_pipeline()

    print(">>> Submitting pipeline to Azure ML ...")
    submitted = ml_client.jobs.create_or_update(pipeline_job)

    print(f">>> Pipeline submitted successfully!")
    print(f"    Job name  : {submitted.name}")
    print(f"    Status    : {submitted.status}")
    print(f"    Studio URL: {submitted.studio_url}")
    print(f"\n>>> Track progress at:")
    print(f"    {submitted.studio_url}")