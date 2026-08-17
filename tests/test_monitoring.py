"""
Tests for monitoring and seasonality analysis
==============================================
Covers the three defects these modules were fixed for:
  1. batches ordered by month name only, interleaving 1994/1995/1996
  2. KS tests with no multiple-comparison correction
  3. drift tested on the batching key itself (Year) and on an ID (RepNumber)
plus the sample-size guard in the seasonality metrics.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _synthetic_claims(seed=0):
    """Three years x twelve months, shuffled — mirrors fraud_oracle's shape."""
    rng = np.random.default_rng(seed)
    rows = []
    for year in (1994, 1995, 1996):
        for m_i, month in enumerate(MONTHS, start=1):
            for _ in range(12):
                rows.append({
                    "Year": year, "Month": month, "_expected_order": year * 100 + m_i,
                    "Age": int(rng.integers(18, 70)),
                    "Deductible": int(rng.choice([300, 400, 500, 700])),
                    "RepNumber": int(rng.integers(1, 17)),
                    "DriverRating": int(rng.integers(1, 5)),
                    "WeekOfMonth": int(rng.integers(1, 6)),
                })
    df = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


class TestBatchOrdering:

    def test_batches_are_chronological_across_years(self):
        from monitoring import create_batches
        df = _synthetic_claims()
        scores = np.random.default_rng(1).random(len(df))
        y = np.zeros(len(df), dtype=int)

        batches = create_batches(df, scores, y, n_batches=6)

        # Every batch's claims must be >= the previous batch's, on (Year, Month)
        last_max = -np.inf
        for b in batches:
            order = b["df"]["_expected_order"].values
            assert order.min() >= last_max, "batch overlaps an earlier period"
            last_max = order.max()

    def test_batches_do_not_mix_all_three_years(self):
        """The bug: sorting by month name alone put 1994/95/96 in every batch."""
        from monitoring import create_batches
        df = _synthetic_claims()
        scores = np.random.default_rng(2).random(len(df))
        batches = create_batches(df, scores, np.zeros(len(df), dtype=int), n_batches=6)

        assert not all(b["df"]["Year"].nunique() == 3 for b in batches)

    def test_all_rows_preserved(self):
        from monitoring import create_batches
        df = _synthetic_claims()
        batches = create_batches(df, np.random.default_rng(3).random(len(df)),
                                 np.zeros(len(df), dtype=int), n_batches=6)
        assert sum(b["n"] for b in batches) == len(df)


class TestDriftFeatureExclusion:

    def test_year_and_repnumber_excluded_by_default(self):
        from monitoring import DRIFT_EXCLUDE
        assert "Year" in DRIFT_EXCLUDE       # batching key — circular to test
        assert "RepNumber" in DRIFT_EXCLUDE  # staff identifier, not a distribution

    def test_excluded_features_are_not_tested(self):
        from monitoring import _manual_drift_check
        df = _synthetic_claims()
        ref, cur = df.iloc[:200], df.iloc[200:]
        result = _manual_drift_check(
            ref, cur,
            feature_cols=["Age", "Deductible", "DriverRating"],   # Year/Rep omitted
            batch_label="t",
        )
        assert "Year" not in result["tested_features"]
        assert "RepNumber" not in result["tested_features"]


class TestMultipleComparisonCorrection:

    def test_identical_distributions_report_no_drift(self):
        """Reference and current drawn from the same data — must be stable."""
        from monitoring import _manual_drift_check
        rng = np.random.default_rng(7)
        base = pd.DataFrame({f"f{i}": rng.normal(size=400) for i in range(7)})
        result = _manual_drift_check(base.iloc[:200], base.iloc[200:],
                                     feature_cols=list(base.columns),
                                     batch_label="same")
        assert result["dataset_drift"] is False
        assert result["n_drifted_features"] == 0

    def test_correction_is_no_looser_than_raw(self):
        from monitoring import _manual_drift_check
        rng = np.random.default_rng(11)
        ref = pd.DataFrame({f"f{i}": rng.normal(size=300) for i in range(8)})
        cur = pd.DataFrame({f"f{i}": rng.normal(size=300) for i in range(8)})
        cur["f0"] = cur["f0"] + 3.0          # one genuine shift

        result = _manual_drift_check(ref, cur, feature_cols=list(ref.columns),
                                      batch_label="shifted")
        # BH can only ever flag the same or fewer features than raw p < 0.05
        assert result["n_drifted_features"] <= result["n_drifted_uncorrected"]

    def test_real_shift_is_still_detected(self):
        """Correction must not be so strict that a true signal is lost."""
        from monitoring import _manual_drift_check
        rng = np.random.default_rng(13)
        ref = pd.DataFrame({f"f{i}": rng.normal(size=300) for i in range(5)})
        cur = pd.DataFrame({f"f{i}": rng.normal(size=300) for i in range(5)})
        cur["f2"] = cur["f2"] + 4.0

        result = _manual_drift_check(ref, cur, feature_cols=list(ref.columns),
                                      batch_label="shifted")
        assert result["dataset_drift"] is True
        assert "f2" in result["drifted_features"]


class TestDriftVerdictIsEnvironmentIndependent:
    """
    The drift verdict must come from the KS + Benjamini-Hochberg path whether or
    not Evidently is installed.

    It did not always. Evidently's numbers were returned when it was importable
    and the KS+BH path ran only as a fallback, so the deployed Space (which has
    Evidently) showed 1 drifted feature in three batches while a machine without
    it showed 0 — under a caption claiming BH correction in both cases.
    """

    def _frames(self):
        rng = np.random.default_rng(3)
        ref = pd.DataFrame({f"f{i}": rng.normal(size=300) for i in range(5)})
        cur = pd.DataFrame({f"f{i}": rng.normal(size=300) for i in range(5)})
        return ref, cur

    def test_report_always_carries_the_corrected_fields(self):
        from monitoring import compute_drift_report
        ref, cur = self._frames()
        out = compute_drift_report(ref, cur, feature_cols=list(ref.columns),
                                   batch_label="t", write_html=False)
        # These keys exist only on the KS+BH path; Evidently's branch never had them
        assert "n_drifted_uncorrected" in out
        assert "tested_features" in out
        assert out["n_drifted_features"] <= out["n_drifted_uncorrected"]

    def test_matches_the_manual_check_exactly(self):
        from monitoring import _manual_drift_check, compute_drift_report
        ref, cur = self._frames()
        direct = _manual_drift_check(ref, cur, list(ref.columns), "t")
        viaapi = compute_drift_report(ref, cur, feature_cols=list(ref.columns),
                                      batch_label="t", write_html=False)
        for key in ("dataset_drift", "n_drifted_features",
                    "n_drifted_uncorrected", "n_features", "drift_share"):
            assert viaapi[key] == direct[key], key

    def test_html_generation_never_changes_the_verdict(self):
        from monitoring import compute_drift_report
        ref, cur = self._frames()
        without = compute_drift_report(ref, cur, feature_cols=list(ref.columns),
                                       batch_label="t", write_html=False)
        with_html = compute_drift_report(ref, cur, feature_cols=list(ref.columns),
                                         batch_label="t", write_html=True)
        assert without["dataset_drift"] == with_html["dataset_drift"]
        assert without["n_drifted_features"] == with_html["n_drifted_features"]


class TestSeasonalitySampleGuard:

    def _month(self, n, n_fraud, seed=0):
        rng = np.random.default_rng(seed)
        y = np.zeros(n, dtype=int)
        y[:n_fraud] = 1
        return {"month": "Apr", "month_num": 4, "n": n,
                "risk_scores": rng.random(n), "y_true": y}

    def test_metrics_suppressed_below_threshold(self):
        """April in the real test split has 3 fraud cases — PR-AUC on 3
        positives is noise and must not be plotted as a seasonal signal."""
        from temporal_analysis import month_metrics
        m = month_metrics(self._month(136, 3))
        assert m["sufficient"] is False
        assert m["pr_auc"] is None
        assert m["precision_5pct"] is None
        assert m["n_fraud"] == 3          # sample size still visible

    def test_metrics_present_above_threshold(self):
        from temporal_analysis import month_metrics
        m = month_metrics(self._month(135, 13))
        assert m["sufficient"] is True
        assert m["pr_auc"] is not None
        assert m["n_fraud"] == 13

    def test_summary_ignores_suppressed_months(self):
        from temporal_analysis import get_temporal_summary
        df = pd.DataFrame([
            {"month": "Apr", "month_num": 4, "n": 136, "n_fraud": 3,
             "pr_auc": None, "fraud_rate": 0.02},
            {"month": "May", "month_num": 5, "n": 135, "n_fraud": 13,
             "pr_auc": 0.31, "fraud_rate": 0.10},
        ])
        summary = get_temporal_summary(df)
        assert summary["n_months"] == 2
        assert summary["n_months_scored"] == 1
        assert summary["best_month"] == "May"
        assert summary["worst_month"] == "May"   # not the unscored month


class TestAgeBandDerivation:

    @pytest.mark.parametrize("age,expected", [
        (16, "16 to 17"), (17, "16 to 17"), (18, "18 to 20"), (20, "18 to 20"),
        (21, "21 to 25"), (25, "21 to 25"), (26, "26 to 30"), (30, "26 to 30"),
        (31, "31 to 35"), (35, "31 to 35"), (36, "36 to 40"), (40, "36 to 40"),
        (41, "41 to 50"), (50, "41 to 50"), (51, "51 to 65"), (65, "51 to 65"),
        (66, "over 65"), (80, "over 65"),
    ])
    def test_boundaries(self, age, expected):
        from feature_engineering import derive_age_band
        assert derive_age_band(age) == expected

    def test_every_band_is_a_known_key(self):
        """A band the ordinal map doesn't know would become NaN downstream."""
        from feature_engineering import derive_age_band, AGE_POLICY_HOLDER_MAP
        for age in range(16, 101):
            assert derive_age_band(age) in AGE_POLICY_HOLDER_MAP

    def test_live_scoring_year_is_in_training_range(self):
        from config import LIVE_SCORING_YEAR
        assert 1994 <= LIVE_SCORING_YEAR <= 1996
