from pathlib import Path

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from core.auth import (
    create_access_token,
    email_hmac,
    encrypt_email,
    get_optional_user,
    hash_password,
    verify_password,
)
from core.database import get_db
from core.db_models import User
from core.logger import get_logger
from routes.db_service import save_prediction
from routes.predict_service import (
    MODEL_NAME,
    PassengerInput,
    _preprocess,
    _predict,
    _AGE_MEDIAN,
    _FARE_MEDIAN,
)
from routes.explain_service import (
    ExplainRequest,
    ExplainResponse,
    _preprocess_single,
    _model as _explain_model,
    MODEL_NAME as _EXPLAIN_MODEL_NAME,
    _get_shap_values,
    _run_explain_agent,
    FeatureContribution,
)

log = get_logger("frontend_service")

TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "templates"
templates     = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(prefix="/ui", tags=["UI"])

# ─── Pages ───────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse, summary="Prediction UI")
def index(request: Request, current_user: dict | None = Depends(get_optional_user)):
    log.info("UI page loaded")
    return templates.TemplateResponse(request, "index.html", {
        "model_name":   MODEL_NAME,
        "current_user": current_user,
    })


@router.get("/explain", response_class=HTMLResponse, summary="Explainability UI")
def explain_page(request: Request, current_user: dict | None = Depends(get_optional_user)):
    log.info("Explain UI page loaded")
    return templates.TemplateResponse(request, "explain.html", {
        "model_name":   MODEL_NAME,
        "current_user": current_user,
    })


# ─── Auth pages ──────────────────────────────────────────────────────────────

@router.get("/login", response_class=HTMLResponse, summary="Login page")
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@router.get("/signup", response_class=HTMLResponse, summary="Signup page")
def signup_page(request: Request):
    return templates.TemplateResponse(request, "signup.html", {})


@router.post("/login", response_class=HTMLResponse, summary="Login form (HTMX)")
def ui_login(
    request:  Request,
    email:    str = Form(...),
    password: str = Form(...),
    db:       Session = Depends(get_db),
):
    from routes.auth_service import _get_user_by_email
    user = _get_user_by_email(db, email)
    if not user or not verify_password(password, user.password_hash):
        log.warning("Failed login attempt for email hash=%s", email_hmac(email)[:8])
        return templates.TemplateResponse(request, "partials/auth_error.html", {
            "message": "Invalid email or password.",
        }, status_code=401)

    token = create_access_token(user.id, user.username)
    log.info("UI login success: user_id=%d", user.id)
    response = HTMLResponse(content="", status_code=200,
                            headers={"HX-Redirect": "/ui/"})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response


@router.post("/signup", response_class=HTMLResponse, summary="Signup form (HTMX)")
def ui_signup(
    request:          Request,
    username:         str = Form(...),
    email:            str = Form(...),
    password:         str = Form(...),
    confirm_password: str = Form(...),
    db:               Session = Depends(get_db),
):
    from routes.auth_service import _get_user_by_email
    if password != confirm_password:
        return templates.TemplateResponse(request, "partials/auth_error.html", {
            "message": "Passwords do not match.",
        }, status_code=422)

    if len(password) < 8:
        return templates.TemplateResponse(request, "partials/auth_error.html", {
            "message": "Password must be at least 8 characters.",
        }, status_code=422)

    if db.query(User).filter(User.username == username).first():
        return templates.TemplateResponse(request, "partials/auth_error.html", {
            "message": "Username already taken.",
        }, status_code=409)

    if _get_user_by_email(db, email):
        return templates.TemplateResponse(request, "partials/auth_error.html", {
            "message": "Email already registered.",
        }, status_code=409)

    user = User(
        username=username,
        email_encrypted=encrypt_email(email),
        email_hash=email_hmac(email),
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.username)
    log.info("UI signup success: user_id=%d username=%s", user.id, user.username)
    response = HTMLResponse(content="", status_code=200,
                            headers={"HX-Redirect": "/ui/"})
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        samesite="lax",
        max_age=86400,
    )
    return response


@router.get("/logout", summary="Logout — clears the auth cookie")
def ui_logout():
    response = RedirectResponse(url="/ui/login", status_code=303)
    response.delete_cookie("access_token")
    return response


# ─── HTMX endpoints (return HTML fragments) ──────────────────────────────────

@router.post("/predict/single", response_class=HTMLResponse,
             summary="Single prediction (HTMX)")
def ui_predict_single(
    request: Request,
    sex:  str   = Form(...),
    age:  float = Form(None),
    fare: float = Form(None),
    db: Session = Depends(get_db),
):
    log.info("UI single prediction | sex=%s age=%s fare=%s", sex, age, fare)
    try:
        passenger = PassengerInput(
            sex=sex,
            age=age   if age  is not None else _AGE_MEDIAN,
            fare=fare if fare is not None else _FARE_MEDIAN,
        )
        array        = _preprocess([passenger])
        preds, probs = _predict(array)
    except Exception as exc:
        log.error("UI single prediction error: %s", exc)
        return templates.TemplateResponse(request, "partials/error.html", {
            "message": str(exc),
        })

    save_prediction(db, sex=sex, age=passenger.age, fare=passenger.fare,
                    survived=preds[0], survival_probability=round(probs[0], 4),
                    model_used=MODEL_NAME, source="ui")

    log.info("UI single result | survived=%d prob=%.4f", preds[0], probs[0])
    return templates.TemplateResponse(request, "partials/single_result.html", {
        "survived":             preds[0],
        "survival_probability": probs[0],
        "model_used":           MODEL_NAME,
    })


@router.post("/predict/bulk", response_class=HTMLResponse,
             summary="Bulk prediction (HTMX)")
def ui_predict_bulk(
    request:  Request,
    csv_data: str = Form(...),
    db: Session = Depends(get_db),
):
    log.info("UI bulk prediction request received")
    rows_input     = []
    parsed_display = []

    for line_num, line in enumerate(csv_data.strip().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]

        sex = parts[0].lower() if parts[0] else "male"
        if sex not in ("male", "female"):
            log.warning("Invalid sex value at row %d: %s", line_num, sex)
            return templates.TemplateResponse(request, "partials/error.html", {
                "message": f"Row {line_num}: sex must be 'male' or 'female', got '{sex}'.",
            })

        raw_age  = parts[1] if len(parts) > 1 else ""
        raw_fare = parts[2] if len(parts) > 2 else ""

        try:
            age  = float(raw_age)  if raw_age  else _AGE_MEDIAN
            fare = float(raw_fare) if raw_fare else _FARE_MEDIAN
        except ValueError:
            log.warning("Non-numeric age/fare at row %d", line_num)
            return templates.TemplateResponse(request, "partials/error.html", {
                "message": f"Row {line_num}: age and fare must be numbers.",
            })

        rows_input.append(PassengerInput(sex=sex, age=age, fare=fare))
        parsed_display.append({"sex": sex, "age": raw_age, "fare": raw_fare})

    if not rows_input:
        log.warning("Bulk request had no valid rows")
        return templates.TemplateResponse(request, "partials/error.html", {
            "message": "No valid rows found. Use format: sex,age,fare (one per line).",
        })

    if len(rows_input) > 1000:
        log.warning("Bulk request exceeded 1000 row limit: %d", len(rows_input))
        return templates.TemplateResponse(request, "partials/error.html", {
            "message": "Max 1000 rows per request.",
        })

    try:
        array        = _preprocess(rows_input)
        preds, probs = _predict(array)
    except Exception as exc:
        log.error("UI bulk prediction error: %s", exc)
        return templates.TemplateResponse(request, "partials/error.html", {
            "message": str(exc),
        })

    survived_count = sum(preds)
    survival_rate  = survived_count / len(preds)
    log.info("UI bulk result | records=%d survived=%d rate=%.4f",
             len(preds), survived_count, survival_rate)

    for row, p, prob in zip(rows_input, preds, probs):
        save_prediction(db, sex=row.sex, age=row.age, fare=row.fare,
                        survived=p, survival_probability=round(prob, 4),
                        model_used=MODEL_NAME, source="ui")

    rows_out = [
        {**display, "survived": p, "survival_probability": prob}
        for display, p, prob in zip(parsed_display, preds, probs)
    ]

    return templates.TemplateResponse(request, "partials/bulk_result.html", {
        "rows":               rows_out,
        "total_records":      len(rows_out),
        "survived_count":     survived_count,
        "not_survived_count": len(rows_out) - survived_count,
        "survival_rate":      survival_rate,
        "model_used":         MODEL_NAME,
    })


@router.post("/explain/single", response_class=HTMLResponse,
             summary="Explain single prediction (HTMX)")
def ui_explain_single(
    request: Request,
    sex:  str   = Form(...),
    age:  float = Form(None),
    fare: float = Form(None),
):
    log.info("UI explain | sex=%s age=%s fare=%s", sex, age, fare)
    try:
        age  = age  if age  is not None else _AGE_MEDIAN
        fare = fare if fare is not None else _FARE_MEDIAN

        arr      = _preprocess_single(sex, age, fare)
        survived = int(_explain_model.predict(arr)[0])
        prob     = round(float(_explain_model.predict_proba(arr)[0, 1]), 4)

        shap_data     = _get_shap_values(sex, age, fare)
        contributions = [
            FeatureContribution(
                feature=c["feature"],
                value=c["raw_value"],
                shap_value=c["shap_value"],
                direction=c["direction"],
            )
            for c in shap_data["contributions"]
        ]

        explanation = _run_explain_agent(sex, age, fare, survived, prob)

        result = ExplainResponse(
            survived=survived,
            survival_probability=prob,
            model_used=MODEL_NAME,
            feature_contributions=contributions,
            explanation=explanation,
        )
    except Exception as exc:
        log.error("UI explain error: %s", exc)
        return templates.TemplateResponse(request, "partials/error.html", {
            "message": str(exc),
        })

    log.info("UI explain done | survived=%d prob=%.4f", survived, prob)
    return templates.TemplateResponse(request, "partials/explain_result.html", {
        "result": result,
    })
