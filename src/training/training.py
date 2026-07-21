# training.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint, uniform

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.training.transformers import (
    fix_total_charges_dtype,
    charges_per_month_of_tenure_raw,
    ratio_feature_name,
)

DATA_URL = (
    "https://raw.githubusercontent.com/Twishha-Soni/Telecom-Customer-Churn-Prediction/"
    "refs/heads/main/Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )

    total_charges_pipeline = make_pipeline(
        FunctionTransformer(fix_total_charges_dtype, feature_names_out="one-to-one"),
        SimpleImputer(strategy="constant", fill_value=0),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )

    categorical_pipeline = OneHotEncoder(handle_unknown="ignore")

    ratio_pipeline = FunctionTransformer(
        charges_per_month_of_tenure_raw,
        feature_names_out=ratio_feature_name,
    )

    return ColumnTransformer([
        ("numeric", numeric_pipeline, ["tenure", "MonthlyCharges"]),
        ("total_charges", total_charges_pipeline, ["TotalCharges"]),
        ("categorical", categorical_pipeline, CATEGORICAL_COLS),
        ("ratio", ratio_pipeline, ["tenure", "TotalCharges", "MonthlyCharges"]),
    ], remainder="passthrough")


def main():
    # --- load data ---
    df = pd.read_csv(DATA_URL)

    # --- split BEFORE any preprocessing (train test split earlier, per notebook) ---
    train_set, test_set = train_test_split(
        df, test_size=0.2, stratify=df["Churn"], random_state=42
    )

    X_train = train_set.drop(columns=["Churn", "customerID"])
    y_train = train_set["Churn"]
    X_test = test_set.drop(columns=["Churn", "customerID"])
    y_test = test_set["Churn"]

    # --- build full pipeline: preprocessing + model as ONE object ---
    preprocessing = build_preprocessing_pipeline()
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("clf", RandomForestClassifier(random_state=42)),
    ])

    # --- hyperparameter search (same distributions/settings as notebook) ---
    param_distributions = {
        "clf__n_estimators": randint(100, 500),
        "clf__max_depth": randint(3, 20),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 10),
        "clf__max_features": uniform(0.1, 0.9),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print("Best ROC-AUC (cross-validated):", search.best_score_)
    print("Best hyperparameters:", search.best_params_)

    # --- evaluate on held-out test set ---
    best_pipeline = search.best_estimator_
    test_accuracy = best_pipeline.score(X_test, y_test)
    print("Test accuracy:", test_accuracy)

    # --- persist the ENTIRE fitted pipeline to disk ---
    project_root = Path(__file__).resolve().parents[2]  # src/training/training.py -> project root
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    output_path = models_dir / "churn_pipeline.joblib"
    joblib.dump(best_pipeline, output_path)
    print(f"Pipeline saved to {output_path}")


if __name__ == "__main__":
    main()