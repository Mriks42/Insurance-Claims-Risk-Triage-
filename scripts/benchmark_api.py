"""
Latency benchmark for the scoring API
======================================
Measures p50 / p95 / p99 for single and batch scoring, with and without SHAP
explanations, plus artifact cold-start time.

    python scripts/benchmark_api.py                      # in-process (no network)
    python scripts/benchmark_api.py --url http://localhost:8000   # live server

In-process mode exercises the whole FastAPI stack — validation, transform,
model, serialisation — minus network transport, so the numbers are the compute
cost of a prediction. Point it at a running server to include transport.

Writes outputs/serving/latency.json.
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join("outputs", "serving")

CLAIM = {
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


def percentiles(samples):
    s = sorted(samples)
    def pct(p):
        idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
        return round(s[idx], 2)
    return {
        "n":      len(s),
        "mean_ms": round(statistics.fmean(s), 2),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "max_ms": round(s[-1], 2),
    }


def measure(call, n, warmup=5):
    for _ in range(warmup):          # exclude first-call costs (lazy explainer,
        call()                       # numpy warm-up, JIT-ish effects)
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        call()
        samples.append((time.perf_counter() - t0) * 1000)
    return percentiles(samples)


def build_client(url):
    if url:
        import httpx
        client = httpx.Client(base_url=url, timeout=30.0)
        return client, "live server: " + url
    from fastapi.testclient import TestClient
    from api.main import app
    client = TestClient(app)
    client.__enter__()               # run lifespan so artifacts load
    return client, "in-process (no network transport)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=None, help="Benchmark a running server")
    ap.add_argument("-n", type=int, default=200, help="Requests per scenario")
    args = ap.parse_args()

    # Cold start: how long loading model + preprocessor + bundle takes. This is
    # the delay before a fresh container can serve its first request.
    from data_pipeline import load_serving_artifacts
    t0 = time.perf_counter()
    load_serving_artifacts()
    cold_start_ms = (time.perf_counter() - t0) * 1000

    client, mode = build_client(args.url)
    print(f"Benchmarking {mode}\n")

    # Reference point: the model on its own, on an already-transformed row.
    # Without this the end-to-end numbers are uninterpretable — it turns out
    # almost none of a request is spent in XGBoost.
    model_only = {}
    if not args.url:
        import numpy as np
        import pandas as pd
        from api.main import to_raw_row
        from api.schemas import Claim
        from data_pipeline import load_serving_artifacts, score_claims_frame

        art = load_serving_artifacts()
        row, _ = to_raw_row(Claim(**CLAIM))
        _, X_one = score_claims_frame(pd.DataFrame([row]), art)
        _, X_hundred = score_claims_frame(pd.DataFrame([row] * 100), art)
        model = art["model"]
        model_only = {
            "predict_1_row":   measure(lambda: model.predict_proba(X_one), 200),
            "predict_100_rows": measure(lambda: model.predict_proba(X_hundred), 200),
        }
        print(f"  {'model only (1 row)':18} p50={model_only['predict_1_row']['p50_ms']:>7.2f}ms")
        print(f"  {'model only (100)':18} p50={model_only['predict_100_rows']['p50_ms']:>7.2f}ms")

    scenarios = {
        "single":              lambda: client.post("/predict", json=CLAIM),
        "single_with_shap":    lambda: client.post("/predict?explain=true", json=CLAIM),
        "batch_10":            lambda: client.post("/predict/batch",
                                                   json={"claims": [CLAIM] * 10}),
        "batch_100":           lambda: client.post("/predict/batch",
                                                   json={"claims": [CLAIM] * 100}),
        "health":              lambda: client.get("/health"),
    }

    results = {}
    for name, call in scenarios.items():
        assert call().status_code == 200, f"{name} did not return 200"
        stats = measure(call, args.n if "batch_100" not in name else max(20, args.n // 5))
        if name.startswith("batch_"):
            size = int(name.split("_")[1])
            stats["per_claim_ms"] = round(stats["p50_ms"] / size, 3)
            stats["throughput_claims_per_s"] = round(size / (stats["p50_ms"] / 1000), 1)
        else:
            stats["throughput_claims_per_s"] = round(1000 / stats["p50_ms"], 1)
        results[name] = stats
        print(f"  {name:18} p50={stats['p50_ms']:>7.2f}ms  p95={stats['p95_ms']:>7.2f}ms  "
              f"p99={stats['p99_ms']:>7.2f}ms  {stats['throughput_claims_per_s']:>8.1f} claims/s")

    payload = {
        "mode":          mode,
        "cold_start_ms": round(cold_start_ms, 1),
        "requests_per_scenario": args.n,
        "model_only":    model_only,
        "scenarios":     results,
        "measured_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "latency.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\ncold start (load artifacts): {cold_start_ms:.1f} ms")
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
