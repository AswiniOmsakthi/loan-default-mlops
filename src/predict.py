"""
predict.py
----------
Loan Default Prediction — loads model and exposes predict functions.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict

# ── Paths ──────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "model.joblib")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")

CLASS_NAMES = ["no_default", "default"]


# ── Load model ─────────────────────────────────────────────────
def _load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run python src/train.py first."
        )
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Features not found at {FEATURES_PATH}. Run python src/train.py first."
        )
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        features = json.load(f)
    return model, features


_model, _features = _load_artifacts()
ALL_FEATURES = _features["all_features"]


# ── Predict single ─────────────────────────────────────────────
def predict(input_data: Dict) -> dict:
    """
    Predict loan default for a single applicant.

    Parameters
    ----------
    input_data : dict with all feature values

    Returns
    -------
    dict with predicted_class, risk_label, default_probability, risk_level
    """
    df = pd.DataFrame([input_data])[ALL_FEATURES]

    pred_class = int(_model.predict(df)[0])
    pred_proba = _model.predict_proba(df)[0]
    default_prob = round(float(pred_proba[1]), 4)

    if default_prob < 0.3:
        risk_level = "LOW"
    elif default_prob < 0.6:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    return {
        "predicted_class":    pred_class,
        "risk_label":         CLASS_NAMES[pred_class],
        "default_probability": default_prob,
        "no_default_probability": round(float(pred_proba[0]), 4),
        "risk_level":         risk_level,
    }


# ── Predict batch ──────────────────────────────────────────────
def predict_batch(samples: List[Dict]) -> List[dict]:
    """
    Predict loan default for multiple applicants.

    Parameters
    ----------
    samples : list of dicts, each with all feature values

    Returns
    -------
    list of prediction dicts
    """
    if not samples:
        raise ValueError("samples list must not be empty.")

    df = pd.DataFrame(samples)[ALL_FEATURES]
    pred_classes = _model.predict(df).tolist()
    pred_probas  = _model.predict_proba(df)

    results = []
    for pred_class, proba_row in zip(pred_classes, pred_probas):
        default_prob = round(float(proba_row[1]), 4)
        if default_prob < 0.3:
            risk_level = "LOW"
        elif default_prob < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        results.append({
            "predicted_class":         int(pred_class),
            "risk_label":              CLASS_NAMES[pred_class],
            "default_probability":     default_prob,
            "no_default_probability":  round(float(proba_row[0]), 4),
            "risk_level":              risk_level,
        })
    return results
