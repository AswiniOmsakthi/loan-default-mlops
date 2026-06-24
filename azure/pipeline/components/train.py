"""
train.py
Pipeline Step 2 — extracted from src/train.py → preprocess() + train_model() + save_model()
"""
import argparse
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection   import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.ensemble          import GradientBoostingClassifier
from sklearn.impute            import SimpleImputer
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import OneHotEncoder


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

# ── 2. Preprocess ──────────────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for numeric + categorical features."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric",      numeric_pipeline,      NUMERIC_FEATURES),
        ("categorical",  categorical_pipeline,  CATEGORICAL_FEATURES),
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data",   type=str)
    parser.add_argument("--output_model", type=str)
    args = parser.parse_args()

    # ── Load CSV saved by data_prep.py ──────────────────────────
    csv_path = os.path.join(args.input_data, "loan_data.csv")
    df = pd.read_csv(csv_path)
    print(f">>> Loaded data → shape: {df.shape}")

    # ── Preprocess (your preprocess() function) ──────────────────
    print(">>> Preprocessing ...")
    
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"    Train : {X_train.shape[0]} samples")
    print(f"    Test  : {X_test.shape[0]} samples")
    print(f"    Default rate train : {y_train.mean():.2%}")
    print(f"    Default rate test  : {y_test.mean():.2%}\n")

    print(">>> Building pipeline ...")
    preprocessor = build_preprocessor()

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=20,
            random_state=RANDOM_STATE,
        )),
    ])

    print(">>> Training GradientBoostingClassifier ...")
    model_pipeline.fit(X_train, y_train)

    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"    CV ROC-AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # ── Save model + scaler (your save_model() function) ─────────
    os.makedirs(args.output_model, exist_ok=True)
    joblib.dump(model_pipeline,  os.path.join(args.output_model, "model.joblib"))
    print(f">>> Model  saved → {args.output_model}/model.joblib")
    

if __name__ == "__main__":
    main()