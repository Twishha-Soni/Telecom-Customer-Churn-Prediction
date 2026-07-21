# Telecom Customer Churn Prediction

## Steps in preprocessing data before training the model

1. Basic exploration of the dataset like tracking the data types of the features available.
2. Stratified train/test split — stratify directly on the Churn label itself and keep the test set away for evaluation at the end.
3. Full visualization pass — look for correlations, combined/derived attributes worth engineering.
4. Separate features (X) from label (y) in train set.
5. Clean data using SimpleImputer.
6. Handle categorical/text columns with OneHotEncoder.
7. Feature scaling — standard scaling for most numeric features, but for any heavy long-tailed distribution (e.g. TotalCharges, MonthlyCharges if skewed), consider an RBF-kernel similarity transform instead of plain scaling or simple log transformation of the heavy long tail data.
8. Custom transformers where useful (FunctionTransformer / a custom class with fit/transform).
9. Combine all of the above into a single sklearn Pipeline (ColumnTransformer + custom transformers).
10. Select 3 candidate models, train each, evaluate on a validation set (not test set yet) — this is a first-pass comparison, not final tuning. Then tune your model using hyperparameter tuning for the model you think would predict more accurately on the data.

## Setup

1. `pip install -r requirements.txt`
2. `python -m src.training.training` — trains the model and saves it to `models/`
3. `uvicorn src.api.main:app --reload`
4. Visit `http://127.0.0.1:8000/`