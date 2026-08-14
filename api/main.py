"""
Fraud triage scoring API
========================
Serves the trained model over HTTP, independently of the Streamlit dashboard.

Run:
    uvicorn api.main:app --reload --port 8000
    open http://localhost:8000/docs

The service loads only the artifacts in outputs/improvement/ — the trained
model, the fitted preprocessor and the serving bundle. It does NOT read
fraud_oracle.csv. That is the point: scoring must not depend on the training
dataset being present, and the transform applied here must be the exact object
fitted during training rather than one rebuilt from data that may have moved.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    BatchRequest, BatchResponse, Claim, HealthResponse,
    PredictionResponse, ReasonCode,
)
from config import LIVE_SCORING_YEAR
from data_pipeline import (
    ArtifactsMissing, assign_bucket_from_thresholds,
    load_serving_artifacts, score_claims_frame,
)
from feature_engineering import derive_age_band, humanize_feature

MAX_BATCH = int(os.environ.get("MAX_BATCH", "500"))

# Populated at startup. Kept module-level so the model, the preprocessor and the
# SHAP explainer are each built once rather than per request.
STATE: Dict[str, Any] = {"artifacts": None, "explainer": None, "error": None}


def load_artifacts() -> None:
    """Load the serving artifacts into module state. Idempotent."""
    if STATE["artifacts"] is not None:
        return
    try:
        STATE["artifacts"] = load_serving_artifacts()
        STATE["error"] = None
    except ArtifactsMissing as exc:            # fail loudly, never score without them
        STATE["artifacts"] = None
        STATE["error"] = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()          # eager: a broken deployment fails at boot, not on
    yield                     # the first customer request
    STATE.update(artifacts=None, explainer=None)


app = FastAPI(
    title="Insurance Claims Fraud Triage API",
    description=(
        "Scores an automotive insurance claim for fraud risk and routes it to "
        "SIU, Manual Review or Approve. Explanations are SHAP-based and opt-in."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _artifacts() -> Dict[str, Any]:
    # Lazy retry as well as the eager load above: a test client that does not run
    # the lifespan still gets a working service rather than a spurious 503.
    load_artifacts()
    if STATE["artifacts"] is None:
        raise HTTPException(status_code=503, detail=STATE["error"] or "artifacts not loaded")
    return STATE["artifacts"]


def _explainer():
    """Build the SHAP explainer once, on first use — it is not needed unless a
    caller asks for explanations, and constructing it costs more than a score."""
    if STATE["explainer"] is None:
        import shap
        STATE["explainer"] = shap.TreeExplainer(_artifacts()["model"])
    return STATE["explainer"]


# ──────────────────────────────────────────────────────────────
# Claim -> raw dataset row
# ──────────────────────────────────────────────────────────────
def to_raw_row(claim: Claim):
    """
    Turn a validated request into a row in the raw dataset's own layout, filling
    the fields the caller may omit. Returns (row_dict, defaults_applied).

    AgeOfPolicyHolder is DERIVED from Age rather than defaulted to a constant —
    the dashboard used to pin it, which meant a 70-year-old was scored with a
    policyholder band of 28.
    """
    row = claim.model_dump()
    defaults: Dict[str, Any] = {}

    fixed = {
        "Make": "Honda",
        "RepNumber": 5,
        "DriverRating": 3,
        "WeekOfMonth": 2,
        "WeekOfMonthClaimed": 2,
        "NumberOfCars": "1 vehicle",
        # Training data spans 1994-1996 and Year is a model feature, so a
        # present-day year would be outside everything the model has seen.
        "Year": LIVE_SCORING_YEAR,
    }
    for field, value in fixed.items():
        if row.get(field) is None:
            row[field] = value
            defaults[field] = value

    if row.get("PolicyType") is None:
        row["PolicyType"] = f"{claim.VehicleCategory} - {claim.BasePolicy}"
        defaults["PolicyType"] = row["PolicyType"]

    if row.get("AgeOfPolicyHolder") is None:
        row["AgeOfPolicyHolder"] = derive_age_band(claim.Age)
        defaults["AgeOfPolicyHolder"] = row["AgeOfPolicyHolder"]

    row["PolicyNumber"] = 0        # dropped before modelling; present for schema parity
    return row, defaults


def _reason_codes(shap_row, X_row, feature_names, cat_cols, top_n=5) -> List[ReasonCode]:
    order = np.argsort(np.abs(shap_row))[::-1][:top_n]
    codes = []
    for i in order:
        sv = float(shap_row[i])
        strength = abs(sv)
        codes.append(ReasonCode(
            label=humanize_feature(feature_names[i], cat_cols, X_row[i]),
            technical=feature_names[i],
            shap=round(sv, 4),
            direction="High risk" if sv > 0 else "Low risk",
            intensity=("STRONG" if strength > 0.5
                       else "MODERATE" if strength > 0.2 else "MILD"),
        ))
    return codes


def _score(claims: List[Claim], explain: bool) -> List[PredictionResponse]:
    art    = _artifacts()
    bundle = art["bundle"]

    rows, defaults = zip(*(to_raw_row(c) for c in claims))
    prob, X_t = score_claims_frame(pd.DataFrame(list(rows)), art)

    shap_values = None
    if explain:
        dense = X_t.toarray() if hasattr(X_t, "toarray") else np.asarray(X_t)
        shap_values = _explainer().shap_values(dense)

    out = []
    for i, p in enumerate(prob):
        codes = None
        if explain:
            dense_row = X_t[i].toarray().ravel() if hasattr(X_t[i], "toarray") \
                else np.asarray(X_t[i]).ravel()
            codes = _reason_codes(shap_values[i], dense_row,
                                  bundle["feature_names"], bundle["cat_cols"])
        out.append(PredictionResponse(
            risk_score=round(float(p), 6),
            triage_bucket=assign_bucket_from_thresholds(float(p), bundle["thresholds"]),
            model_version=bundle["model_version"],
            thresholds=bundle["thresholds"],
            defaults_applied=defaults[i],
            reason_codes=codes,
        ))
    return out


# ──────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness plus whether the model artifacts actually loaded."""
    loaded = STATE["artifacts"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        artifacts_loaded=loaded,
        model_version=(STATE["artifacts"]["bundle"]["model_version"] if loaded else None),
        detail=STATE["error"],
    )


@app.get("/model", tags=["ops"])
def model_info():
    """
    What is being served: version, when it was trained, held-out performance with
    its confidence interval, and the bucket cut-points the router applies.
    """
    bundle = _artifacts()["bundle"]
    return {
        "model_version":    bundle["model_version"],
        "model_name":       bundle["model_name"],
        "trained_at":       bundle["trained_at"],
        "n_features":       bundle["n_features"],
        "metrics":          bundle["metrics"],
        "thresholds":       bundle["thresholds"],
        "cost_assumptions": bundle["cost_assumptions"],
        "max_batch":        MAX_BATCH,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(claim: Claim, explain: bool = False):
    """Score a single claim. Pass `?explain=true` for SHAP reason codes."""
    return _score([claim], explain)[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(request: BatchRequest):
    """Score many claims in one transform pass."""
    if not request.claims:
        raise HTTPException(status_code=422, detail="claims must not be empty")
    if len(request.claims) > MAX_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(request.claims)} exceeds MAX_BATCH={MAX_BATCH}",
        )
    preds = _score(request.claims, request.explain)
    return BatchResponse(predictions=preds, count=len(preds))


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Server-side latency, so benchmarks can separate compute from transport."""
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


@app.exception_handler(ArtifactsMissing)
async def artifacts_missing_handler(request: Request, exc: ArtifactsMissing):
    return JSONResponse(status_code=503, content={"detail": str(exc)})
