"""
generate_data.py
----------------
Generates a realistic Loan Default dataset.
Run: python src/generate_data.py
Saves: data/loan_data.csv
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES    = 10000

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

def generate_loan_data(n_samples: int = N_SAMPLES) -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)

    # ── Demographics ───────────────────────────────────────────
    age              = np.random.randint(21, 65, n_samples)
    gender           = np.random.choice(["Male", "Female"], n_samples, p=[0.55, 0.45])
    education        = np.random.choice(
                         ["High School", "Bachelor", "Master", "PhD"],
                         n_samples, p=[0.30, 0.45, 0.20, 0.05]
                       )
    employment_type  = np.random.choice(
                         ["Salaried", "Self-Employed", "Business", "Unemployed"],
                         n_samples, p=[0.55, 0.25, 0.15, 0.05]
                       )
    years_employed   = np.clip(np.random.normal(7, 4, n_samples), 0, 40).astype(int)

    # ── Financial ──────────────────────────────────────────────
    annual_income    = np.clip(
                         np.random.lognormal(11.5, 0.6, n_samples),
                         150000, 5000000
                       ).astype(int)
    loan_amount      = np.clip(
                         annual_income * np.random.uniform(0.5, 5.0, n_samples),
                         50000, 10000000
                       ).astype(int)
    loan_term_months = np.random.choice([12, 24, 36, 48, 60, 84, 120], n_samples)
    interest_rate    = np.round(np.random.uniform(7.5, 24.0, n_samples), 2)
    credit_score     = np.clip(np.random.normal(680, 80, n_samples), 300, 900).astype(int)
    existing_loans   = np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.40, 0.30, 0.18, 0.08, 0.04])
    monthly_emi      = np.round(
                         loan_amount * (interest_rate / 1200) /
                         (1 - (1 + interest_rate / 1200) ** (-loan_term_months)),
                         2
                       )
    emi_to_income    = np.round((monthly_emi * 12) / annual_income, 4)
    loan_to_income   = np.round(loan_amount / annual_income, 4)

    # ── Loan purpose ───────────────────────────────────────────
    loan_purpose     = np.random.choice(
                         ["Home", "Education", "Vehicle", "Business", "Personal", "Medical"],
                         n_samples, p=[0.30, 0.15, 0.20, 0.15, 0.12, 0.08]
                       )

    # ── Default label (realistic rules) ────────────────────────
    default_prob = (
        0.05
        + 0.25 * (credit_score < 600)
        + 0.15 * (emi_to_income > 0.5)
        + 0.10 * (loan_to_income > 4)
        + 0.12 * (employment_type == "Unemployed")
        + 0.08 * (existing_loans >= 3)
        + 0.06 * (interest_rate > 18)
        - 0.05 * (credit_score > 750)
        - 0.04 * (education == "PhD")
        - 0.03 * (employment_type == "Salaried")
    )
    default_prob = np.clip(default_prob, 0.02, 0.90)
    default      = (np.random.random(n_samples) < default_prob).astype(int)

    # ── Introduce missing values (realistic) ───────────────────
    df = pd.DataFrame({
        "age":              age,
        "gender":           gender,
        "education":        education,
        "employment_type":  employment_type,
        "years_employed":   years_employed,
        "annual_income":    annual_income,
        "loan_amount":      loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate":    interest_rate,
        "credit_score":     credit_score,
        "existing_loans":   existing_loans,
        "monthly_emi":      monthly_emi,
        "emi_to_income":    emi_to_income,
        "loan_to_income":   loan_to_income,
        "loan_purpose":     loan_purpose,
        "default":          default,
    })

    # Add 3% missing values to some columns
    for col in ["credit_score", "years_employed", "annual_income"]:
        mask = np.random.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    print(">>> Generating loan dataset ...")
    df = generate_loan_data()

    csv_path = os.path.join(DATA_DIR, "loan_data.csv")
    df.to_csv(csv_path, index=False)

    print(f"    Saved         → {csv_path}")
    print(f"    Shape         → {df.shape}")
    print(f"    Default rate  → {df['default'].mean():.2%}")
    print(f"    Missing vals  → {df.isnull().sum().sum()}")
    print(f"\n    Feature types:")
    print(df.dtypes.to_string())
