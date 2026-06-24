"""
app.py
------
Loan Default Prediction — FastAPI REST API.
Run: uvicorn src.app:app --host 0.0.0.0 --port 8000
"""

import json
import os
import time
from typing import List, Optional
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from predict import predict, predict_batch, ALL_FEATURES

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Loan Default Prediction API",
    description=(
        "Predicts whether a loan applicant will default "
        "using a GradientBoosting model trained on banking data."
    ),
    version="1.0.0",
)


# ── Enums ──────────────────────────────────────────────────────
class GenderEnum(str, Enum):
    male   = "Male"
    female = "Female"

class EducationEnum(str, Enum):
    high_school = "High School"
    bachelor    = "Bachelor"
    master      = "Master"
    phd         = "PhD"

class EmploymentEnum(str, Enum):
    salaried      = "Salaried"
    self_employed = "Self-Employed"
    business      = "Business"
    unemployed    = "Unemployed"

class LoanPurposeEnum(str, Enum):
    home      = "Home"
    education = "Education"
    vehicle   = "Vehicle"
    business  = "Business"
    personal  = "Personal"
    medical   = "Medical"


# ── Request schema ─────────────────────────────────────────────
class LoanApplication(BaseModel):
    age:              int        = Field(..., ge=18, le=80,       example=35,           description="Applicant age")
    gender:           GenderEnum = Field(...,                      example="Male",       description="Gender")
    education:        EducationEnum = Field(...,                   example="Bachelor",   description="Education level")
    employment_type:  EmploymentEnum = Field(...,                  example="Salaried",   description="Employment type")
    years_employed:   int        = Field(..., ge=0, le=45,        example=8,            description="Years at current job")
    annual_income:    float      = Field(..., gt=0,               example=750000,       description="Annual income in INR")
    loan_amount:      float      = Field(..., gt=0,               example=2000000,      description="Loan amount in INR")
    loan_term_months: int        = Field(..., ge=6, le=360,       example=60,           description="Loan term in months")
    interest_rate:    float      = Field(..., ge=1.0, le=36.0,    example=11.5,         description="Annual interest rate %")
    credit_score:     float      = Field(..., ge=300, le=900,     example=720,          description="Credit score")
    existing_loans:   int        = Field(..., ge=0, le=10,        example=1,            description="Number of existing loans")
    monthly_emi:      float      = Field(..., gt=0,               example=42000,        description="Monthly EMI in INR")
    emi_to_income:    float      = Field(..., ge=0, le=1,         example=0.67,         description="Annual EMI / Annual income")
    loan_to_income:   float      = Field(..., ge=0,               example=2.67,         description="Loan amount / Annual income")
    loan_purpose:     LoanPurposeEnum = Field(...,                example="Home",       description="Purpose of loan")

    def to_dict(self) -> dict:
        return {
            "age":              self.age,
            "gender":           self.gender.value,
            "education":        self.education.value,
            "employment_type":  self.employment_type.value,
            "years_employed":   self.years_employed,
            "annual_income":    self.annual_income,
            "loan_amount":      self.loan_amount,
            "loan_term_months": self.loan_term_months,
            "interest_rate":    self.interest_rate,
            "credit_score":     self.credit_score,
            "existing_loans":   self.existing_loans,
            "monthly_emi":      self.monthly_emi,
            "emi_to_income":    self.emi_to_income,
            "loan_to_income":   self.loan_to_income,
            "loan_purpose":     self.loan_purpose.value,
        }


class BatchLoanApplication(BaseModel):
    applications: List[LoanApplication] = Field(..., min_length=1, max_length=100)


# ── Response schema ────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class:          int
    risk_label:               str
    default_probability:      float
    no_default_probability:   float
    risk_level:               str
    input_features:           dict
    latency_ms:               float
    decision:                 str

class BatchPredictionResponse(BaseModel):
    predictions:   List[dict]
    total:         int
    high_risk:     int
    medium_risk:   int
    low_risk:      int
    latency_ms:    float


# ── Routes ─────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "message": "Loan Default Prediction API is running.",
        "docs":    "/docs",
        "health":  "/health",
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":    "healthy",
        "model":     "GradientBoostingClassifier",
        "version":   "1.0.0",
        "use_case":  "Loan Default Prediction",
    }


@app.get("/metadata", tags=["Info"])
def metadata():
    metrics_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "metrics.json"
    )
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
    return {
        "features":    ALL_FEATURES,
        "classes":     ["no_default", "default"],
        "risk_levels": ["LOW", "MEDIUM", "HIGH"],
        "metrics":     metrics,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(application: LoanApplication):
    """
    Predict loan default risk for a **single** applicant.

    Risk levels:
    - **LOW** — default probability < 30%
    - **MEDIUM** — default probability 30-60%
    - **HIGH** — default probability > 60%
    """
    t0 = time.perf_counter()
    try:
        result = predict(application.to_dict())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = round((time.perf_counter() - t0) * 1000, 3)
    decision   = "REJECT" if result["default_probability"] >= 0.5 else "APPROVE"

    return PredictionResponse(
        **result,
        input_features=application.to_dict(),
        latency_ms=latency_ms,
        decision=decision,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch_endpoint(batch: BatchLoanApplication):
    """
    Predict loan default risk for a **batch** of up to 100 applicants.
    """
    t0 = time.perf_counter()
    samples = [a.to_dict() for a in batch.applications]
    try:
        results = predict_batch(samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms  = round((time.perf_counter() - t0) * 1000, 3)
    high_risk   = sum(1 for r in results if r["risk_level"] == "HIGH")
    medium_risk = sum(1 for r in results if r["risk_level"] == "MEDIUM")
    low_risk    = sum(1 for r in results if r["risk_level"] == "LOW")

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        high_risk=high_risk,
        medium_risk=medium_risk,
        low_risk=low_risk,
        latency_ms=latency_ms,
    )
