import numpy as np
import pandas as pd


def fix_total_charges_dtype(X):
    fixed = pd.to_numeric(np.asarray(X).ravel(), errors="coerce")
    return fixed.reshape(-1, 1)


def charges_per_month_of_tenure_raw(X):
    X = np.asarray(X, dtype=object)
    tenure = X[:, 0].astype(float)
    total_charges = pd.to_numeric(X[:, 1], errors="coerce")
    monthly_charges = X[:, 2].astype(float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = total_charges / tenure

    # zero-tenure customers: no billing history to divide by,
    # fall back to their MonthlyCharges rate
    ratio = np.where(tenure == 0, monthly_charges, ratio)
    return ratio.reshape(-1, 1)


def ratio_feature_name(transformer, input_features):
    return ["charges_per_month_of_tenure"]