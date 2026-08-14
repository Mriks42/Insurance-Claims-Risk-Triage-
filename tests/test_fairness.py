"""
Tests for fairness reporting
=============================
Covers two defects that were visible on the dashboard:
  1. a group that is never flagged became the ratio denominator, driving every
     other group's disparate impact to 0.000 and flagging them all as "Concern"
  2. rows with Age == 0 (not a real age) fell outside every bin and rendered as
     a demographic group literally labelled "nan"
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _analysis(rows):
    """rows: list of (group, flagged, fraud) -> frame group_metrics expects."""
    return pd.DataFrame([
        {"grp": g, "flagged": f, "actual_fraud": y, "risk_score": 0.5 * f}
        for g, f, y in rows
    ])


class TestDisparateImpactReference:

    def test_never_flagged_group_does_not_zero_the_column(self):
        """The bug: 65+ had flag_rate 0.0, so min/x made every group 0.000."""
        from fairness_analysis import group_metrics
        rows = ([("young", 1, 0)] * 30 + [("young", 0, 0)] * 70 +   # 30% flagged
                [("mid",   1, 0)] * 15 + [("mid",   0, 0)] * 85 +   # 15% flagged
                [("old",   0, 0)] * 50)                             # never flagged
        out = group_metrics(_analysis(rows), "grp")

        di = dict(zip(out["group"], out["disparate_impact"]))
        assert pd.isna(di["old"]), "never-flagged group has no defined ratio"
        # reference is the least-flagged group WITH flags: mid at 0.15
        assert di["mid"] == 1.0
        assert abs(di["young"] - 0.5) < 1e-6
        assert not (out["disparate_impact"].fillna(1) == 0).any()

    def test_never_flagged_group_is_labelled_not_passed(self):
        """NaN must not fall through to the '✅ OK' branch."""
        from fairness_analysis import group_metrics
        rows = ([("a", 1, 0)] * 20 + [("a", 0, 0)] * 80 +
                [("b", 0, 0)] * 50)
        out = group_metrics(_analysis(rows), "grp")
        flag = dict(zip(out["group"], out["di_flag"]))
        assert "never flagged" in flag["b"]
        assert "OK" not in flag["b"]

    def test_equal_flag_rates_are_all_ok(self):
        from fairness_analysis import group_metrics
        rows = ([("a", 1, 0)] * 20 + [("a", 0, 0)] * 80 +
                [("b", 1, 0)] * 20 + [("b", 0, 0)] * 80)
        out = group_metrics(_analysis(rows), "grp")
        assert (out["disparate_impact"] == 1.0).all()
        assert out["di_flag"].str.contains("OK").all()

    def test_below_threshold_is_flagged(self):
        from fairness_analysis import group_metrics
        rows = ([("a", 1, 0)] * 40 + [("a", 0, 0)] * 60 +   # 0.40
                [("b", 1, 0)] * 10 + [("b", 0, 0)] * 90)    # 0.10 -> di 0.25
        out = group_metrics(_analysis(rows), "grp")
        di = dict(zip(out["group"], out["disparate_impact"]))
        assert abs(di["a"] - 0.25) < 1e-6
        assert "Concern" in out.set_index("group").loc["a", "di_flag"]


class TestAgeBinning:

    def test_age_zero_is_labelled_not_nan(self):
        from fairness_analysis import bin_age, AGE_UNKNOWN_LABEL
        out = bin_age(pd.Series([0, 0, 22, 40, 70]))
        assert out.isna().sum() == 0, "no NaN should survive binning"
        assert (out == AGE_UNKNOWN_LABEL).sum() == 2
        assert str(out.iloc[2]) == "16-25"
        assert str(out.iloc[4]) == "65+"

    def test_no_group_renders_as_the_string_nan(self):
        from fairness_analysis import bin_age
        out = bin_age(pd.Series([0, 16, 30, 55, 90])).astype(str)
        assert "nan" not in set(out)

    def test_real_ages_bin_unchanged(self):
        from fairness_analysis import bin_age
        out = bin_age(pd.Series([16, 25, 26, 35, 36, 50, 51, 65, 66])).astype(str)
        assert list(out) == ["16-25", "16-25", "26-35", "26-35",
                             "36-50", "36-50", "51-65", "51-65", "65+"]
