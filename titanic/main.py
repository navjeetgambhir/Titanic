import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from core.database import init_db
from routes.predict_service import router as predict_router
from routes.frontend_service import router as frontend_router
from routes.db_service import router as db_router
from routes.explain_service import router as explain_router
from routes.auth_service import router as auth_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predict Titanic passenger survival using Sex, Age, and Fare.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(auth_router)
app.include_router(predict_router)
app.include_router(frontend_router)
app.include_router(db_router)
app.include_router(explain_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
