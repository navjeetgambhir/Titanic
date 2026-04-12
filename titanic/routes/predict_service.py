import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from core.database import get_db
from core.logger import get_logger

log = get_logger("predict_service")

# ─── Load model & scaler at startup ──────────────────────────────────────────

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

def _load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

_model_files = list(MODELS_DIR.glob("*_model.pkl"))
if not _model_files:
    raise RuntimeError(f"No model pickle found in {MODELS_DIR}")

MODEL_PATH  = _model_files[0]
MODEL_NAME  = MODEL_PATH.stem
SCALER_PATH = MODELS_DIR / "scaler.pkl"

model  = _load_pickle(MODEL_PATH)
scaler = _load_pickle(SCALER_PATH) if SCALER_PATH.exists() else None

log.info("Loaded model: %s", MODEL_PATH.name)
log.info("Scaler loaded: %s", scaler is not None)

router = APIRouter(prefix="/predict", tags=["Predictions"])

# ─── Schemas ─────────────────────────────────────────────────────────────────

class PassengerInput(BaseModel):
    sex:  Literal["male", "female"] = Field(..., json_schema_extra={"example": "female"})
    age:  float                     = Field(..., ge=0, le=120, json_schema_extra={"example": 29.0})
    fare: float                     = Field(..., ge=0, json_schema_extra={"example": 32.20})

    @field_validator("age", "fare", mode="before")
    @classmethod
    def coerce_none_to_median(cls, v):
        return v

class SinglePredictionResponse(BaseModel):
    survived:             int
    survival_probability: float
    model_used:           str

class BulkPredictionResponse(BaseModel):
    predictions:   list[SinglePredictionResponse]
    total_records: int
    survival_rate: float

# ─── Helpers ─────────────────────────────────────────────────────────────────

_AGE_MEDIAN  = 28.0
_FARE_MEDIAN = 14.45

def _preprocess(records: list[PassengerInput]) -> np.ndarray:
    df = pd.DataFrame([{
        "Sex":  1 if r.sex == "female" else 0,
        "Age":  r.age  if r.age  is not None else _AGE_MEDIAN,
        "Fare": r.fare if r.fare is not None else _FARE_MEDIAN,
    } for r in records])
    return scaler.transform(df) if scaler else df.values

def _predict(array: np.ndarray) -> tuple[list[int], list[float]]:
    preds = model.predict(array).tolist()
    probs = model.predict_proba(array)[:, 1].tolist()
    return preds, probs

# ─── Routes ──────────────────────────────────────────────────────────────────

@router.post("/single", response_model=SinglePredictionResponse,
             summary="Predict survival for one passenger")
def predict_single(passenger: PassengerInput, db: Session = Depends(get_db)):
    """Pass **one** passenger's Sex, Age, and Fare. Returns survival prediction and probability."""
    log.info("Single prediction | sex=%s age=%s fare=%s", passenger.sex, passenger.age, passenger.fare)
    try:
        array        = _preprocess([passenger])
        preds, probs = _predict(array)
    except Exception as e:
        log.error("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # persist to DB
    from routes.db_service import save_prediction
    save_prediction(db, sex=passenger.sex, age=passenger.age, fare=passenger.fare,
                    survived=preds[0], survival_probability=round(probs[0], 4),
                    model_used=MODEL_NAME, source="api")

    log.info("Single result | survived=%d prob=%.4f", preds[0], probs[0])
    return SinglePredictionResponse(
        survived=preds[0],
        survival_probability=round(probs[0], 4),
        model_used=MODEL_NAME,
    )


@router.post("/bulk", response_model=BulkPredictionResponse,
             summary="Predict survival for multiple passengers")
def predict_bulk(passengers: list[PassengerInput], db: Session = Depends(get_db)):
    """Pass a **list** of passengers. Returns per-passenger predictions and aggregate survival rate."""
    if not passengers:
        raise HTTPException(status_code=422, detail="Passenger list must not be empty.")
    if len(passengers) > 1000:
        raise HTTPException(status_code=422, detail="Max 1000 records per request.")

    log.info("Bulk prediction | records=%d", len(passengers))
    try:
        array        = _preprocess(passengers)
        preds, probs = _predict(array)
    except Exception as e:
        log.error("Bulk prediction error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # persist all records
    from routes.db_service import save_prediction
    for passenger, pred, prob in zip(passengers, preds, probs):
        save_prediction(db, sex=passenger.sex, age=passenger.age, fare=passenger.fare,
                        survived=pred, survival_probability=round(prob, 4),
                        model_used=MODEL_NAME, source="api")

    survival_rate = round(sum(preds) / len(preds), 4)
    log.info("Bulk result | records=%d survival_rate=%.4f", len(preds), survival_rate)

    results = [
        SinglePredictionResponse(
            survived=p,
            survival_probability=round(prob, 4),
            model_used=MODEL_NAME,
        )
        for p, prob in zip(preds, probs)
    ]
    return BulkPredictionResponse(
        predictions=results,
        total_records=len(results),
        survival_rate=survival_rate,
    )
