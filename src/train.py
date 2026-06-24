"""
train.py
--------
Loan Default Prediction — full ML pipeline.
Run: python src/train.py
"""

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
from sklearn.metrics           import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(ROOT_DIR, "data")
MODELS_DIR   = os.path.join(ROOT_DIR, "models")

DATA_PATH    = os.path.join(DATA_DIR,   "loan_data.csv")
MODEL_PATH   = os.path.join(MODELS_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
FEATURES_PATH= os.path.join(MODELS_DIR, "features.json")

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


# ── 1. Load Data ───────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print(">>> Loading loan dataset ...")
    df = pd.read_csv(DATA_PATH)
    print(f"    Shape        : {df.shape}")
    print(f"    Default rate : {df[TARGET].mean():.2%}")
    print(f"    Missing vals : {df.isnull().sum().sum()}\n")
    return df


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


# ── 3. Train ───────────────────────────────────────────────────
def train(df: pd.DataFrame) -> tuple:
    print(">>> Splitting data ...")
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

    return model_pipeline, X_test, y_test


# ── 4. Evaluate ────────────────────────────────────────────────
def evaluate(model_pipeline, X_test, y_test) -> dict:
    print(">>> Evaluating model ...")
    y_pred      = model_pipeline.predict(X_test)
    y_pred_prob = model_pipeline.predict_proba(X_test)[:, 1]

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

    # Feature importances (from classifier step)
    classifier      = model_pipeline.named_steps["classifier"]
    preprocessor    = model_pipeline.named_steps["preprocessor"]
    cat_encoder     = preprocessor.named_transformers_["categorical"].named_steps["encoder"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names   = NUMERIC_FEATURES + cat_feature_names
    importances     = dict(zip(
        feature_names,
        classifier.feature_importances_.round(4).tolist()
    ))
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

    metrics = {
        "accuracy":               round(float(accuracy),  4),
        "precision":              round(float(precision), 4),
        "recall":                 round(float(recall),    4),
        "f1_score":               round(float(f1),        4),
        "roc_auc":                round(float(roc_auc),   4),
        "classification_report":  report,
        "confusion_matrix":       conf_matrix,
        "top_10_features":        top_features,
        "test_size":              len(y_test),
        "default_rate_test":      round(float(y_test.mean()), 4),
    }

    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Default', 'Default'])}")
    print(f"    Confusion matrix:\n{np.array(conf_matrix)}\n")

    return metrics


# ── 5. Save ────────────────────────────────────────────────────
def save(model_pipeline, metrics: dict) -> None:
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f">>> Model saved   → {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f">>> Metrics saved → {METRICS_PATH}")

    features_info = {
        "numeric_features":      NUMERIC_FEATURES,
        "categorical_features":  CATEGORICAL_FEATURES,
        "all_features":          ALL_FEATURES,
        "target":                TARGET,
    }
    with open(FEATURES_PATH, "w") as f:
        json.dump(features_info, f, indent=2)
    print(f">>> Features saved → {FEATURES_PATH}\n")


# ── Main ───────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATA_DIR,   exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    df                         = load_data()
    model_pipeline, X_test, y_test = train(df)
    metrics                    = evaluate(model_pipeline, X_test, y_test)
    save(model_pipeline, metrics)

    print("=" * 60)
    print(f"  Training complete.")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print("=" * 60)
