"""
evaluate.py
Pipeline Step 3 — extracted from src/train.py → evaluate_model()
"""
import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics           import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ── Config ─────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
TARGET       = "default"

NUMERIC_FEATURES = [
    "age", "years_employed", "annual_income",
    "loan_amount", "loan_term_months", "interest_rate",
    "credit_score", "existing_loans", "monthly_emi",
    "emi_to_income", "loan_to_income",
]

CATEGORICAL_FEATURES = [
    "gender", "education", "employment_type", "loan_purpose"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",     type=str)
    parser.add_argument("--input_model",    type=str)
    parser.add_argument("--output_metrics", type=str)
    args = parser.parse_args()

    # ── Load data ────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(args.input_data, "loan_data.csv"))
    X  = df[ALL_FEATURES]
    y  = df[TARGET]

    _, X_test, _, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ── Load model + preprocessor ─────────────────────────────────
    model  = joblib.load(os.path.join(args.input_model, "model.joblib"))

    # ── Evaluate (your evaluate_model() function) ────────────────
    print(">>> Evaluating model ...")
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    accuracy    = accuracy_score(y_test, y_pred)
    precision   = precision_score(y_test, y_pred)
    recall      = recall_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_pred_prob)
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()
    report      = classification_report(
                    y_test, y_pred,
                    target_names=["No Default", "Default"],
                    output_dict=True,
                  )
    

    metrics = {
        "accuracy":               round(float(accuracy),  4),
        "precision":              round(float(precision), 4),
        "recall":                 round(float(recall),    4),
        "f1_score":               round(float(f1),        4),
        "roc_auc":                round(float(roc_auc),   4),
        "classification_report":  report,
        "confusion_matrix":       conf_matrix,
        "test_size":              len(y_test),
        "default_rate_test":      round(float(y_test.mean()), 4),
    }

    # ── Save metrics.json ────────────────────────────────────────
    os.makedirs(args.output_metrics, exist_ok=True)
    metrics_path = os.path.join(args.output_metrics, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default', 'Default'])}")
    print(f"    Confusion matrix:\n{np.array(conf_matrix)}\n")

if __name__ == "__main__":
    main()