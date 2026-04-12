# Titanic Survival Predictor

A FastAPI application that predicts Titanic passenger survival using a trained ML model, with SHAP-powered explainability driven by a GPT-4o mini agent.

---

## Features

- **Survival prediction** — single and bulk endpoints using Sex, Age, and Fare
- **AI explainability** — GPT-4o mini agent uses SHAP values, historical statistics, and counterfactuals to generate plain-English explanations
- **JWT authentication** — signup/login with Argon2 password hashing and Fernet-encrypted email storage
- **Prediction history** — all predictions persisted to SQLite via SQLAlchemy
- **Web UI** — Jinja2-templated frontend served at `/ui/`
- **Containerised** — Docker + Docker Compose for local dev; CI/CD to Azure Container Apps via GitHub Actions

---

## Project Structure

```
titanic/
├── main.py                  # FastAPI app entry point
├── core/
│   ├── auth.py              # JWT, hashing, encryption helpers
│   ├── database.py          # SQLAlchemy engine and session
│   ├── db_models.py         # ORM models (User, Prediction)
│   └── logger.py            # Structured logging
├── routes/
│   ├── auth_service.py      # POST /auth/signup, POST /auth/login
│   ├── predict_service.py   # POST /predict/single, POST /predict/bulk
│   ├── explain_service.py   # POST /explain/single (SHAP + GPT agent)
│   ├── db_service.py        # GET /predictions
│   └── frontend_service.py  # GET /ui/
├── models/
│   ├── logistic_regression_model.pkl
│   └── scaler.pkl
├── templates/               # Jinja2 HTML templates
├── tests/                   # pytest test suite
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/deploy.yml
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/signup` | Register a new user |
| `POST` | `/auth/login` | Log in, receive JWT |
| `POST` | `/predict/single` | Predict survival for one passenger |
| `POST` | `/predict/bulk` | Predict survival for up to 1000 passengers |
| `POST` | `/explain/single` | Predict + plain-English explanation via GPT-4o mini |
| `GET`  | `/predictions` | Retrieve stored prediction history |
| `GET`  | `/ui/` | Web interface |
| `GET`  | `/docs` | Auto-generated Swagger UI |

### Example — single prediction

```bash
curl -X POST http://localhost:8000/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sex": "female", "age": 29, "fare": 32.20}'
```

```json
{
  "survived": 1,
  "survival_probability": 0.9123,
  "model_used": "logistic_regression_model"
}
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for `/explain`) | OpenAI key for the GPT-4o mini explainability agent |
| `JWT_SECRET` | Yes | Secret key used to sign JWTs |
| `FERNET_KEY` | Yes | Fernet key for encrypting stored emails |
| `HMAC_SECRET` | Yes | HMAC key for email lookup |
| `DB_PATH` | No | SQLite database path (default: `titanic.db`) |

---

## Running Locally

### With Docker Compose (recommended)

1. Copy and fill in the environment file:
   ```bash
   cp .env.example .env   # edit with your keys
   ```

2. Start the app:
   ```bash
   docker compose up --build
   ```

3. Open [http://localhost:8000/ui/](http://localhost:8000/ui/) or the API docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Without Docker

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
# Install dependencies
uv sync

# Set environment variables
export OPENAI_API_KEY=...
export JWT_SECRET=...
export FERNET_KEY=...
export HMAC_SECRET=...

# Run
uv run python main.py
```

---

## Running Tests

```bash
uv run pytest -q
```

---

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for the full step-by-step guide to deploying on Azure Container Apps.

The GitHub Actions pipeline (`.github/workflows/deploy.yml`) automatically:
1. Runs the test suite
2. Builds and pushes a Docker image to Azure Container Registry
3. Deploys the new image to Azure Container Apps

Deployments are triggered on every push to `main`.

---

## How the Explainability Agent Works

`POST /explain/single` runs a multi-step agentic loop:

1. GPT-4o mini is given the passenger details and three tools:
   - `get_shap_values` — computes per-feature SHAP contributions
   - `get_feature_statistics` — returns historical Titanic survival rates by subgroup
   - `get_counterfactual` — finds the smallest input change that flips the prediction
2. The model calls the tools autonomously to gather evidence
3. It synthesises a 3–5 sentence plain-English explanation

Requires `OPENAI_API_KEY` to be set; returns a fallback message if it is missing.
