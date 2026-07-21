# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

app = FastAPI()

# --- load the trained pipeline ONCE at startup, not per-request ---
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / "models" / "churn_pipeline.joblib"

if not model_path.exists():
    raise RuntimeError(
        f"No trained model found at {model_path}. "
        f"Run 'python -m src.training.training' first."
    )

pipeline = joblib.load(model_path)

# --- schema: the explicit contract for req/res shape ---
# Field names match the raw columns the pipeline was trained on.
class CustomerRequest(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    # accepted as a string since the raw dataset stores it that way
    # (blank string for zero-tenure customers); the pipeline's own
    # fix_total_charges_dtype step handles the numeric coercion
    TotalCharges: str


class ChurnResponse(BaseModel):
    prediction: str
    probabilities: dict[str, float]


@app.get("/")
def serve_frontend():
    frontend_path = project_root / "src" / "frontend" / "index.html"
    return FileResponse(frontend_path)


@app.post("/predict", response_model=ChurnResponse)
def predict(request: CustomerRequest):
    row = pd.DataFrame([request.model_dump()])

    proba = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_
    probabilities = {str(cls): float(p) for cls, p in zip(classes, proba)}

    prediction = pipeline.predict(row)[0]

    return ChurnResponse(prediction=prediction, probabilities=probabilities)
