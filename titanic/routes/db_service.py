from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database import get_db
from core.db_models import SurvivalPrediction
from core.logger import get_logger

log = get_logger("db_service")

router = APIRouter(prefix="/predictions", tags=["Database"])


# ─── Pydantic response schema ─────────────────────────────────────────────────

class PredictionRecord(BaseModel):
    id:                   int
    sex:                  str
    age:                  float
    fare:                 float
    survived:             int
    survival_probability: float
    model_used:           str
    source:               str
    created_at:           datetime

    model_config = {"from_attributes": True}


class PredictionListResponse(BaseModel):
    total:   int
    records: list[PredictionRecord]


# ─── Internal helper used by predict_service ─────────────────────────────────

def save_prediction(
    db: Session,
    *,
    sex: str,
    age: float,
    fare: float,
    survived: int,
    survival_probability: float,
    model_used: str,
    source: str = "api",
) -> SurvivalPrediction:
    record = SurvivalPrediction(
        sex=sex,
        age=age,
        fare=fare,
        survived=survived,
        survival_probability=survival_probability,
        model_used=model_used,
        source=source,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    log.debug("Saved prediction id=%d survived=%d source=%s", record.id, survived, source)
    return record


# ─── Routes ──────────────────────────────────────────────────────────────────

@router.get("/", response_model=PredictionListResponse,
            summary="List all stored predictions")
def list_predictions(
    skip:     int            = Query(0,    ge=0,   description="Records to skip"),
    limit:    int            = Query(100,  ge=1, le=1000, description="Max records to return"),
    survived: Optional[int]  = Query(None, description="Filter by outcome: 0 or 1"),
    source:   Optional[str]  = Query(None, description="Filter by source: 'api' or 'ui'"),
    db: Session = Depends(get_db),
):
    log.info("List predictions | skip=%d limit=%d survived=%s source=%s",
             skip, limit, survived, source)
    q = db.query(SurvivalPrediction)
    if survived is not None:
        q = q.filter(SurvivalPrediction.survived == survived)
    if source:
        q = q.filter(SurvivalPrediction.source == source)
    total   = q.count()
    records = q.order_by(SurvivalPrediction.created_at.desc()).offset(skip).limit(limit).all()
    return PredictionListResponse(total=total, records=records)


@router.get("/{prediction_id}", response_model=PredictionRecord,
            summary="Get a single prediction by ID")
def get_prediction(prediction_id: int, db: Session = Depends(get_db)):
    log.info("Get prediction id=%d", prediction_id)
    record = db.get(SurvivalPrediction, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found.")
    return record


@router.delete("/{prediction_id}", status_code=204,
               summary="Delete a prediction by ID")
def delete_prediction(prediction_id: int, db: Session = Depends(get_db)):
    log.info("Delete prediction id=%d", prediction_id)
    record = db.get(SurvivalPrediction, prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Prediction {prediction_id} not found.")
    db.delete(record)
    db.commit()


@router.get("/stats/summary", response_model=dict,
            summary="Aggregate stats across all stored predictions")
def prediction_stats(db: Session = Depends(get_db)):
    total     = db.query(SurvivalPrediction).count()
    survived  = db.query(SurvivalPrediction).filter(SurvivalPrediction.survived == 1).count()
    api_count = db.query(SurvivalPrediction).filter(SurvivalPrediction.source == "api").count()
    ui_count  = db.query(SurvivalPrediction).filter(SurvivalPrediction.source == "ui").count()

    log.info("Stats requested | total=%d survived=%d", total, survived)
    return {
        "total_predictions":  total,
        "survived":           survived,
        "not_survived":       total - survived,
        "survival_rate":      round(survived / total, 4) if total else 0.0,
        "by_source":          {"api": api_count, "ui": ui_count},
    }
