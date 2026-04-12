"""
Explainability agent: given a passenger + prediction, uses SHAP feature
contributions as tool outputs and OpenAI GPT-4o mini as the reasoning engine
to produce a plain-English survival explanation.
"""
import json
import os
import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from openai import OpenAI
from core.logger import get_logger

log = get_logger("explain_service")

# ─── Load the same model/scaler used by predict_service ─────────────────────

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


_model_files = list(MODELS_DIR.glob("*_model.pkl"))
if not _model_files:
    raise RuntimeError(f"No model pickle found in {MODELS_DIR}")

MODEL_PATH = _model_files[0]
MODEL_NAME = MODEL_PATH.stem
SCALER_PATH = MODELS_DIR / "scaler.pkl"

_model = _load_pickle(MODEL_PATH)
_scaler = _load_pickle(SCALER_PATH) if SCALER_PATH.exists() else None

# ─── SHAP explainer (created once at import time) ────────────────────────────

_FEATURE_NAMES = ["Sex", "Age", "Fare"]
_BACKGROUND = np.array([
    [1, 29.0, 14.45],   # female, median
    [0, 29.0, 14.45],   # male, median
    [1, 10.0,  7.00],
    [0, 45.0, 30.00],
])

if _scaler:
    _BACKGROUND_SCALED = _scaler.transform(_BACKGROUND)
else:
    _BACKGROUND_SCALED = _BACKGROUND

try:
    model_type = type(_model).__name__.lower()
    if "logistic" in model_type:
        _explainer = shap.LinearExplainer(_model, _BACKGROUND_SCALED,
                                          feature_names=_FEATURE_NAMES)
    else:
        _explainer = shap.TreeExplainer(_model, feature_names=_FEATURE_NAMES)
except Exception as e:
    log.warning("SHAP explainer init failed (%s); falling back to KernelExplainer", e)
    _explainer = shap.KernelExplainer(
        _model.predict_proba,
        _BACKGROUND_SCALED,
        feature_names=_FEATURE_NAMES,
    )

log.info("SHAP explainer ready: %s", type(_explainer).__name__)

# ─── Training-set statistics (hard-coded from Titanic train set) ─────────────

_FEATURE_STATS = {
    "Sex": {
        "description": "Passenger gender encoded as 1=female, 0=male",
        "female_survival_rate": 0.742,
        "male_survival_rate":   0.189,
    },
    "Age": {
        "description": "Passenger age in years",
        "mean": 29.7,
        "median": 28.0,
        "min": 0.42,
        "max": 80.0,
        "survival_rate_under_18": 0.54,
        "survival_rate_18_to_60": 0.38,
        "survival_rate_over_60":  0.27,
    },
    "Fare": {
        "description": "Ticket price in British pounds",
        "mean": 32.2,
        "median": 14.45,
        "min": 0.0,
        "max": 512.33,
        "survival_rate_low_fare":  0.30,
        "survival_rate_high_fare": 0.60,
    },
}

# ─── Router ──────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/explain", tags=["Explainability"])

# ─── Schemas ─────────────────────────────────────────────────────────────────


class ExplainRequest(BaseModel):
    sex:  Literal["male", "female"] = Field(..., json_schema_extra={"example": "female"})
    age:  float = Field(..., ge=0, le=120, json_schema_extra={"example": 29.0})
    fare: float = Field(..., ge=0, json_schema_extra={"example": 32.20})


class FeatureContribution(BaseModel):
    feature: str
    value:   Any
    shap_value: float
    direction: str       # "increases_survival" | "decreases_survival" | "neutral"


class ExplainResponse(BaseModel):
    survived:             int
    survival_probability: float
    model_used:           str
    feature_contributions: list[FeatureContribution]
    explanation:          str


# ─── Internal helpers ─────────────────────────────────────────────────────────


_AGE_MEDIAN  = 28.0
_FARE_MEDIAN = 14.45


def _preprocess_single(sex: str, age: float, fare: float) -> np.ndarray:
    arr = np.array([[1 if sex == "female" else 0, age, fare]], dtype=float)
    return _scaler.transform(arr) if _scaler else arr


def _get_shap_values(sex: str, age: float, fare: float) -> dict:
    """Compute SHAP values for one passenger. Returns feature-level contributions."""
    arr = _preprocess_single(sex, age, fare)
    sv = _explainer.shap_values(arr)

    # LinearExplainer and TreeExplainer return different shapes
    if isinstance(sv, list):
        sv = sv[1]          # class-1 (survived) for binary classifiers
    if sv.ndim == 2:
        sv = sv[0]

    raw = {"Sex": float(sv[0]), "Age": float(sv[1]), "Fare": float(sv[2])}

    def _direction(v):
        if v > 0.01:
            return "increases_survival"
        if v < -0.01:
            return "decreases_survival"
        return "neutral"

    contributions = [
        {
            "feature": f,
            "raw_value": 1 if (f == "Sex" and sex == "female") else (0 if f == "Sex" else (age if f == "Age" else fare)),
            "shap_value": round(raw[f], 4),
            "direction": _direction(raw[f]),
        }
        for f in _FEATURE_NAMES
    ]
    return {
        "contributions": contributions,
        "base_value": round(float(_explainer.expected_value if not isinstance(_explainer.expected_value, np.ndarray)
                                  else _explainer.expected_value[1]), 4),
    }


def _get_feature_statistics(feature: str) -> dict:
    """Return training-set statistics for a given feature."""
    if feature not in _FEATURE_STATS:
        return {"error": f"Unknown feature '{feature}'. Valid: {list(_FEATURE_STATS.keys())}"}
    return _FEATURE_STATS[feature]


def _get_counterfactual(sex: str, age: float, fare: float) -> dict:
    """Show what changes would flip the prediction."""
    arr = _preprocess_single(sex, age, fare)
    orig_prob = float(_model.predict_proba(arr)[0, 1])
    orig_pred = int(orig_prob >= 0.5)

    flips = []
    # Try flipping sex
    other_sex = "female" if sex == "male" else "male"
    arr2 = _preprocess_single(other_sex, age, fare)
    prob2 = float(_model.predict_proba(arr2)[0, 1])
    if int(prob2 >= 0.5) != orig_pred:
        flips.append({"change": f"sex → {other_sex}", "new_probability": round(prob2, 4)})

    # Try age extremes
    for new_age in [5, 18, 40, 65]:
        if abs(new_age - age) < 5:
            continue
        arr2 = _preprocess_single(sex, new_age, fare)
        prob2 = float(_model.predict_proba(arr2)[0, 1])
        if int(prob2 >= 0.5) != orig_pred:
            flips.append({"change": f"age → {new_age}", "new_probability": round(prob2, 4)})
            break

    # Try fare extremes
    for new_fare in [5, 50, 200]:
        if abs(new_fare - fare) < 10:
            continue
        arr2 = _preprocess_single(sex, age, new_fare)
        prob2 = float(_model.predict_proba(arr2)[0, 1])
        if int(prob2 >= 0.5) != orig_pred:
            flips.append({"change": f"fare → {new_fare}", "new_probability": round(prob2, 4)})
            break

    return {
        "original_prediction": orig_pred,
        "original_probability": round(orig_prob, 4),
        "flips_that_change_outcome": flips or [{"change": "No simple single-feature flip found."}],
    }


# ─── GPT-4o mini agentic loop ────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_shap_values",
            "description": (
                "Compute SHAP feature contributions for the given passenger. "
                "Returns per-feature SHAP values showing how much each feature pushed "
                "the prediction up or down relative to the base rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sex":  {"type": "string", "enum": ["male", "female"]},
                    "age":  {"type": "number"},
                    "fare": {"type": "number"},
                },
                "required": ["sex", "age", "fare"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_statistics",
            "description": (
                "Return Titanic training-set statistics for a feature (Sex, Age, or Fare) "
                "including survival rates by subgroup. Use this to give historical context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string", "enum": ["Sex", "Age", "Fare"]},
                },
                "required": ["feature"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_counterfactual",
            "description": (
                "Find the smallest input changes that would flip the survival prediction. "
                "Useful for explaining 'what would have had to be different?'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sex":  {"type": "string", "enum": ["male", "female"]},
                    "age":  {"type": "number"},
                    "fare": {"type": "number"},
                },
                "required": ["sex", "age", "fare"],
            },
        },
    },
]

_TOOL_MAP = {
    "get_shap_values":        lambda inp: _get_shap_values(**inp),
    "get_feature_statistics": lambda inp: _get_feature_statistics(inp["feature"]),
    "get_counterfactual":     lambda inp: _get_counterfactual(**inp),
}

_SYSTEM_PROMPT = """You are an expert machine-learning explainability assistant for a Titanic
survival prediction model trained on Sex, Age, and Fare.

Your job:
1. Use get_shap_values to understand which features drove this specific prediction.
2. Use get_feature_statistics for any feature whose contribution is large or surprising.
3. Use get_counterfactual to show what inputs would have changed the outcome.
4. Write a clear, empathetic, non-technical explanation (3-5 sentences) that:
   - States whether the passenger is predicted to survive.
   - Names the most important factor(s) with their direction of influence.
   - Gives one piece of historical context (e.g. overall survival rates by gender).
   - Mentions the counterfactual if it is illuminating.
   Avoid jargon like "SHAP value" in the final explanation — use plain language."""


def _run_explain_agent(sex: str, age: float, fare: float,
                       survived: int, survival_probability: float) -> str:
    """Run the GPT-4o mini agentic loop and return the plain-text explanation."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "Explainability agent unavailable: OPENAI_API_KEY is not set. "
            "Set this environment variable to enable natural-language explanations."
        )

    client = OpenAI(api_key=api_key)
    user_msg = (
        f"Please explain why the model {'predicted survival' if survived else 'predicted death'} "
        f"for this Titanic passenger:\n"
        f"  Sex: {sex}\n"
        f"  Age: {age}\n"
        f"  Fare: {fare}\n"
        f"  Survival probability: {survival_probability:.1%}\n\n"
        f"Use your tools to gather evidence, then write a plain-English explanation."
    )

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    max_iterations = 10
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=_TOOLS,
            messages=messages,
        )

        msg = response.choices[0].message
        # Append the assistant turn (must include tool_calls if present)
        messages.append(msg)

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            return msg.content.strip() if msg.content else "No explanation generated."

        if finish_reason == "tool_calls":
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    result = _TOOL_MAP[tc.function.name](args)
                except Exception as exc:
                    result = {"error": str(exc)}
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      json.dumps(result),
                })
            continue

        # Unexpected finish reason
        break

    return "Explanation could not be generated (agent loop exceeded iteration limit)."


# ─── Endpoint ────────────────────────────────────────────────────────────────


@router.post("/single", response_model=ExplainResponse,
             summary="Explain a survival prediction for one passenger")
def explain_single(passenger: ExplainRequest):
    """
    Predict survival for the passenger **and** return a plain-English explanation
    generated by a GPT-4o mini agent that inspects SHAP feature contributions,
    historical statistics, and counterfactuals.
    """
    log.info("Explain request | sex=%s age=%s fare=%s",
             passenger.sex, passenger.age, passenger.fare)

    arr = _preprocess_single(passenger.sex, passenger.age, passenger.fare)
    survived = int(_model.predict(arr)[0])
    prob     = round(float(_model.predict_proba(arr)[0, 1]), 4)

    # SHAP contributions for structured response
    shap_result = _get_shap_values(passenger.sex, passenger.age, passenger.fare)
    contributions = [
        FeatureContribution(
            feature=c["feature"],
            value=c["raw_value"],
            shap_value=c["shap_value"],
            direction=c["direction"],
        )
        for c in shap_result["contributions"]
    ]

    # Agent explanation
    explanation = _run_explain_agent(
        passenger.sex, passenger.age, passenger.fare, survived, prob
    )

    log.info("Explain done | survived=%d prob=%.4f", survived, prob)
    return ExplainResponse(
        survived=survived,
        survival_probability=prob,
        model_used=MODEL_NAME,
        feature_contributions=contributions,
        explanation=explanation,
    )
