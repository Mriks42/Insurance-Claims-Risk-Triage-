"""
Tests for the scoring API
==========================
The important one is TestTrainServeSkew: it asserts that a claim scored through
the HTTP path gets the same probability as the offline model on the same row.
That is what proves the persisted preprocessor is the transform the model was
trained under, rather than a lookalike rebuilt at request time.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
from fastapi.testclient import TestClient   # noqa: E402

from api.main import app, MAX_BATCH         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_PRESENT = os.path.exists(
    os.path.join(ROOT, "outputs", "improvement", "serving_bundle.json")
)
needs_artifacts = pytest.mark.skipif(
    not ARTIFACTS_PRESENT,
    reason="serving artifacts absent — run `python train.py`",
)

# fraud_oracle.csv is deliberately not in the repo (Kaggle terms), so CI never
# has it. Most of this file needs only the committed artifacts and runs
# everywhere; the skew guard has to compare against the offline model on real
# rows, so it needs the dataset and skips without it.
DATASET_PRESENT = os.path.exists(os.path.join(ROOT, "fraud_oracle.csv"))
needs_dataset = pytest.mark.skipif(
    not DATASET_PRESENT,
    reason="fraud_oracle.csv absent — download from Kaggle to run this locally",
)

VALID_CLAIM = {
    "BasePolicy": "Liability", "VehicleCategory": "Sport",
    "VehiclePrice": "more than 69000", "Fault": "Policy Holder",
    "PoliceReportFiled": "No", "WitnessPresent": "No", "AgentType": "External",
    "AddressChange_Claim": "under 6 months", "Deductible": 400,
    "PastNumberOfClaims": "2 to 4", "NumberOfSuppliments": "more than 5",
    "Age": 23, "Sex": "Male", "MaritalStatus": "Single",
    "AccidentArea": "Urban", "AgeOfVehicle": "7 years",
    "Month": "Jan", "DayOfWeek": "Monday", "Days_Policy_Accident": "none",
    "MonthClaimed": "Jan", "DayOfWeekClaimed": "Tuesday",
    "Days_Policy_Claim": "more than 30",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestOps:

    def test_health_reports_loaded_artifacts(self, client):
        body = client.get("/health").json()
        assert body["status"] in {"ok", "degraded"}
        assert body["artifacts_loaded"] is ARTIFACTS_PRESENT

    @needs_artifacts
    def test_model_endpoint_matches_bundle(self, client):
        import json
        body = client.get("/model").json()
        with open("outputs/improvement/serving_bundle.json") as f:
            bundle = json.load(f)
        assert body["model_version"] == bundle["model_version"]
        assert body["n_features"] == bundle["n_features"]
        assert body["thresholds"] == bundle["thresholds"]
        # A served metric without an interval invites over-reading a point estimate
        assert len(body["metrics"]["test_pr_auc_ci_95"]) == 2

    @needs_artifacts
    def test_thresholds_are_ordered(self, client):
        t = client.get("/model").json()["thresholds"]
        assert t["siu_threshold"] > t["manual_threshold"]


@needs_artifacts
class TestPredict:

    def test_happy_path(self, client):
        r = client.post("/predict", json=VALID_CLAIM)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["risk_score"] <= 1.0
        assert body["triage_bucket"] in {"SIU", "Manual Review", "Approve"}
        assert body["model_version"] == "v1"

    def test_defaults_are_disclosed(self, client):
        """A caller must never be silently scored on values it did not supply."""
        body = client.post("/predict", json=VALID_CLAIM).json()
        applied = body["defaults_applied"]
        assert "Year" in applied and applied["Year"] == 1996
        assert applied["AgeOfPolicyHolder"] == "21 to 25"   # derived from Age=23
        assert applied["PolicyType"] == "Sport - Liability"

    def test_supplied_values_are_not_overridden(self, client):
        payload = {**VALID_CLAIM, "Year": 1995, "Make": "Toyota"}
        body = client.post("/predict", json=payload).json()
        assert "Year" not in body["defaults_applied"]
        assert "Make" not in body["defaults_applied"]

    def test_age_band_tracks_age(self, client):
        young = client.post("/predict", json={**VALID_CLAIM, "Age": 23}).json()
        old   = client.post("/predict", json={**VALID_CLAIM, "Age": 70}).json()
        assert young["defaults_applied"]["AgeOfPolicyHolder"] == "21 to 25"
        assert old["defaults_applied"]["AgeOfPolicyHolder"] == "over 65"

    def test_deterministic(self, client):
        a = client.post("/predict", json=VALID_CLAIM).json()["risk_score"]
        b = client.post("/predict", json=VALID_CLAIM).json()["risk_score"]
        assert a == b

    def test_explanations_opt_in(self, client):
        plain = client.post("/predict", json=VALID_CLAIM).json()
        assert plain["reason_codes"] is None

        explained = client.post("/predict?explain=true", json=VALID_CLAIM).json()
        codes = explained["reason_codes"]
        assert len(codes) == 5
        assert explained["risk_score"] == plain["risk_score"]   # explaining changes nothing
        for c in codes:
            assert c["label"] and c["technical"]
            assert c["direction"] in {"High risk", "Low risk"}
            assert c["intensity"] in {"STRONG", "MODERATE", "MILD"}

    def test_timing_header_present(self, client):
        r = client.post("/predict", json=VALID_CLAIM)
        assert float(r.headers["X-Process-Time-ms"]) >= 0


@needs_artifacts
class TestValidation:

    def test_unknown_category_rejected_with_field_name(self, client):
        r = client.post("/predict", json={**VALID_CLAIM, "BasePolicy": "Comprehensive"})
        assert r.status_code == 422
        assert "BasePolicy" in str(r.json())

    def test_age_zero_rejected(self, client):
        """320 training rows carry Age == 0; it is a data artifact, not an age,
        and must not be accepted as input."""
        r = client.post("/predict", json={**VALID_CLAIM, "Age": 0})
        assert r.status_code == 422
        assert "Age" in str(r.json())

    def test_invalid_deductible_rejected(self, client):
        r = client.post("/predict", json={**VALID_CLAIM, "Deductible": 350})
        assert r.status_code == 422

    def test_missing_required_field_rejected(self, client):
        payload = {k: v for k, v in VALID_CLAIM.items() if k != "BasePolicy"}
        r = client.post("/predict", json=payload)
        assert r.status_code == 422

    def test_unknown_field_rejected(self, client):
        """extra='forbid' — a misspelled field is an error, not a silent default."""
        r = client.post("/predict", json={**VALID_CLAIM, "Deductable": 400})
        assert r.status_code == 422


@needs_artifacts
class TestBatch:

    def test_batch_matches_single(self, client):
        single = client.post("/predict", json=VALID_CLAIM).json()["risk_score"]
        batch = client.post("/predict/batch",
                            json={"claims": [VALID_CLAIM, VALID_CLAIM]}).json()
        assert batch["count"] == 2
        assert all(p["risk_score"] == single for p in batch["predictions"])

    def test_empty_batch_rejected(self, client):
        assert client.post("/predict/batch", json={"claims": []}).status_code == 422

    def test_oversized_batch_rejected(self, client):
        r = client.post("/predict/batch",
                        json={"claims": [VALID_CLAIM] * (MAX_BATCH + 1)})
        assert r.status_code == 413
        assert str(MAX_BATCH) in r.json()["detail"]


@needs_artifacts
class TestTrainServeSkew:
    """The guard that matters: HTTP scoring must equal offline scoring."""

    @needs_dataset
    def test_api_matches_offline_model(self, client):
        from data_pipeline import (build_model_dataset, load_best_model,
                                   raw_rows_for)

        data  = build_model_dataset()
        model = load_best_model()
        offline = model.predict_proba(data["X_test_t"])[:, 1]
        raw = raw_rows_for(data, "test")

        for pos in (0, 7, 250, 1000):
            row = raw.iloc[pos]
            payload = {k: (int(row[k]) if k in ("Deductible", "Age") else row[k])
                       for k in VALID_CLAIM}
            # supply the columns the API would otherwise default, so the
            # comparison isolates the transform rather than the defaults
            payload.update({
                "Make": row["Make"], "RepNumber": int(row["RepNumber"]),
                "DriverRating": int(row["DriverRating"]),
                "WeekOfMonth": int(row["WeekOfMonth"]),
                "WeekOfMonthClaimed": int(row["WeekOfMonthClaimed"]),
                "NumberOfCars": row["NumberOfCars"], "Year": int(row["Year"]),
                "PolicyType": row["PolicyType"],
                "AgeOfPolicyHolder": row["AgeOfPolicyHolder"],
            })
            if payload["Age"] < 16:      # Age==0 rows cannot be expressed via the API
                continue

            served = client.post("/predict", json=payload).json()["risk_score"]
            assert abs(served - float(offline[pos])) < 1e-6, (
                f"train/serve skew at test row {pos}: "
                f"api={served} offline={offline[pos]}"
            )

    def test_bucket_agrees_with_persisted_thresholds(self, client):
        import json
        with open("outputs/improvement/serving_bundle.json") as f:
            thresholds = json.load(f)["thresholds"]
        body = client.post("/predict", json=VALID_CLAIM).json()
        score = body["risk_score"]
        expected = ("SIU" if score >= thresholds["siu_threshold"]
                    else "Manual Review" if score >= thresholds["manual_threshold"]
                    else "Approve")
        assert body["triage_bucket"] == expected
