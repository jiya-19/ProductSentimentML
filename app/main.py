"""
main.py

FastAPI backend for ProductSentimentML.

Serves:
  - GET  /             -> the static frontend (index.html)
  - GET  /health        -> simple health check
  - POST /predict       -> sentiment prediction for a submitted review

Loads the deployable, calibrated scikit-learn Pipeline saved by the notebook
(TF-IDF vectorizer + CalibratedClassifierCV-wrapped Linear SVM, or the
best classical model directly if it already produced genuine
probabilities — see notebooks/Product_Sentiment_Analysis.ipynb, Section 16).
Probabilities returned by this API are genuine calibrated probabilities,
never raw decision-function scores mislabeled as confidence.

Designed to run identically locally and on Render: no machine-specific
paths, no hardcoded port (reads $PORT via the uvicorn start command).
"""

import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

# --- Paths (project-relative, work locally and on Render) ------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"
STATIC_DIR = APP_DIR / "static"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from preprocessing import clean_text  # noqa: E402

MAX_REVIEW_LENGTH = 5000  # characters; guards against excessively long input

app = FastAPI(
    title="ProductSentimentML API",
    description="AI-powered product review sentiment classification (Negative / Neutral / Positive).",
    version="2.0.0",
)

_model = None


def get_model():
    """Lazily load and cache the trained pipeline."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"No trained model found at '{MODEL_PATH}'. Run the notebook "
                "(notebooks/Product_Sentiment_Analysis.ipynb) first to train and save it."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class ReviewRequest(BaseModel):
    review: str = Field(..., description="The product review text to classify.")

    @field_validator("review")
    @classmethod
    def review_must_be_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Review text must not be empty.")
        if len(v) > MAX_REVIEW_LENGTH:
            raise ValueError(f"Review text must be {MAX_REVIEW_LENGTH} characters or fewer.")
        return v


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float
    probabilities: dict


@app.on_event("startup")
def _load_model_on_startup():
    # Fail fast with a clear error at startup rather than on the first request.
    try:
        get_model()
    except RuntimeError as exc:
        # Do not crash the whole app if the model is genuinely missing (e.g. a fresh
        # clone before the notebook has been run) - /predict will report the error.
        print(f"[startup warning] {exc}")


@app.get("/health")
def health():
    model_available = MODEL_PATH.exists()
    return {"status": "ok", "model_loaded": model_available}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: ReviewRequest):
    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    cleaned = clean_text(payload.review)
    if not cleaned:
        raise HTTPException(
            status_code=422,
            detail="Review text contained no usable content after cleaning (e.g. only stopwords/punctuation).",
        )

    clf = model.named_steps["clf"]
    prediction = model.predict([cleaned])[0]

    if not hasattr(clf, "predict_proba"):
        # Should not happen with the deployed calibrated pipeline, but guarded
        # so the API never silently mislabels a decision score as a probability.
        raise HTTPException(
            status_code=500,
            detail="Deployed model does not provide calibrated probabilities.",
        )

    proba = model.predict_proba([cleaned])[0]
    classes = list(clf.classes_)
    probabilities = {cls: round(float(proba[classes.index(cls)]), 4) for cls in classes}
    confidence = probabilities[prediction]

    return PredictionResponse(
        sentiment=prediction,
        confidence=confidence,
        probabilities=probabilities,
    )


# --- Static frontend ---------------------------------------------------
# Mounted last so it does not shadow the /predict and /health API routes above.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def read_index():
    return FileResponse(str(STATIC_DIR / "index.html"))
