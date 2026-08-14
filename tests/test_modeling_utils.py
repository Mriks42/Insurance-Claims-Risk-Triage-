"""
Tests for modeling utility functions
======================================
Tests precision_at_k, recall_at_k, triage bucket assignment,
and ROI calculation logic.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Tests: precision_at_k and recall_at_k
# ============================================================
class TestPrecisionRecallAtK:

    def test_perfect_precision_at_k(self):
        from modeling import precision_at_k
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert precision_at_k(y_true, y_prob, k=3) == 1.0

    def test_zero_precision_at_k(self):
        from modeling import precision_at_k
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert precision_at_k(y_true, y_prob, k=3) == 0.0

    def test_perfect_recall_at_k(self):
        from modeling import recall_at_k
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        assert recall_at_k(y_true, y_prob, k=3) == 1.0

    def test_partial_recall_at_k(self):
        from modeling import recall_at_k
        y_true = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1])
        # Top 3: indices 0,1,2 → 2 fraud out of 3 total fraud
        result = recall_at_k(y_true, y_prob, k=3)
        assert abs(result - 2/3) < 0.001

    def test_precision_at_k_range(self):
        from modeling import precision_at_k
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        result = precision_at_k(y_true, y_prob, k=10)
        assert 0.0 <= result <= 1.0

    def test_recall_at_k_range(self):
        from modeling import recall_at_k
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        result = recall_at_k(y_true, y_prob, k=10)
        assert 0.0 <= result <= 1.0


# ============================================================
# Tests: triage bucket assignment (app.py helper)
# ============================================================
class TestTriageBuckets:

    def setup_method(self):
        """Create a score array for bucket testing."""
        np.random.seed(42)
        self.all_scores = np.random.rand(1000)

    def test_high_score_is_siu(self):
        from app import assign_bucket
        # Score at 99th percentile should be SIU
        high_score = float(np.percentile(self.all_scores, 99))
        bucket = assign_bucket(high_score, all_scores=self.all_scores)
        assert bucket == "SIU"

    def test_low_score_is_approve(self):
        from app import assign_bucket
        # Score at 1st percentile should be Approve
        low_score = float(np.percentile(self.all_scores, 1))
        bucket = assign_bucket(low_score, all_scores=self.all_scores)
        assert bucket == "Approve"

    def test_mid_score_is_manual_review(self):
        from app import assign_bucket
        # Score at 88th percentile should be Manual Review (between 80th and 95th)
        mid_score = float(np.percentile(self.all_scores, 88))
        bucket = assign_bucket(mid_score, all_scores=self.all_scores)
        assert bucket == "Manual Review"

    def test_bucket_values_are_valid(self):
        from app import assign_bucket
        valid_buckets = {"SIU", "Manual Review", "Approve"}
        for score in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            bucket = assign_bucket(score, all_scores=self.all_scores)
            assert bucket in valid_buckets

    def test_siu_count_is_5_percent(self):
        from app import assign_bucket
        buckets = [assign_bucket(s, all_scores=self.all_scores)
                   for s in self.all_scores]
        siu_pct = buckets.count("SIU") / len(buckets)
        assert abs(siu_pct - 0.05) < 0.01

    def test_manual_count_is_15_percent(self):
        from app import assign_bucket
        buckets = [assign_bucket(s, all_scores=self.all_scores)
                   for s in self.all_scores]
        manual_pct = buckets.count("Manual Review") / len(buckets)
        assert abs(manual_pct - 0.15) < 0.01


# ============================================================
# Tests: ROI calculation
# ============================================================
class TestROICalculation:

    def test_roi_keys_present(self):
        from app import compute_test_roi
        np.random.seed(42)
        test_prob = np.random.rand(1000)
        y_test    = np.random.randint(0, 2, 1000)
        roi = compute_test_roi(test_prob, y_test)
        for key in ["fraud_caught", "savings_usd", "costs_usd",
                    "net_benefit", "roi_x"]:
            assert key in roi

    def test_roi_non_negative_savings(self):
        from app import compute_test_roi
        np.random.seed(42)
        test_prob = np.random.rand(500)
        y_test    = np.random.randint(0, 2, 500)
        roi = compute_test_roi(test_prob, y_test)
        assert roi["savings_usd"] >= 0

    def test_roi_costs_positive(self):
        from app import compute_test_roi
        np.random.seed(42)
        test_prob = np.random.rand(500)
        y_test    = np.random.randint(0, 2, 500)
        roi = compute_test_roi(test_prob, y_test)
        assert roi["costs_usd"] > 0

    def test_net_benefit_equals_savings_minus_costs(self):
        from app import compute_test_roi
        np.random.seed(42)
        test_prob = np.random.rand(500)
        y_test    = np.random.randint(0, 2, 500)
        roi = compute_test_roi(test_prob, y_test)
        assert abs(roi["net_benefit"] - (roi["savings_usd"] - roi["costs_usd"])) < 1


# ============================================================
# Tests: reason code generation (app.py helper)
# ============================================================
class TestReasonCodes:

    def test_returns_correct_count(self):
        from app import make_reason_codes
        shap_row  = np.array([0.5, -0.3, 0.1, 0.8, -0.2, 0.05, 0.4, -0.1, 0.3, 0.2])
        feat_names = [f"feature_{i}" for i in range(10)]
        cat_cols   = []
        codes = make_reason_codes(shap_row, feat_names, cat_cols, top_n=5)
        assert len(codes) == 5

    def test_direction_positive_shap(self):
        from app import make_reason_codes
        shap_row   = np.array([0.9, 0.0, 0.0, 0.0, 0.0])
        feat_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        codes = make_reason_codes(shap_row, feat_names, [], top_n=1)
        assert codes[0]["direction"] == "High risk"

    def test_direction_negative_shap(self):
        from app import make_reason_codes
        shap_row   = np.array([-0.9, 0.0, 0.0, 0.0, 0.0])
        feat_names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
        codes = make_reason_codes(shap_row, feat_names, [], top_n=1)
        assert codes[0]["direction"] == "Low risk"

    def test_ohe_decoding(self):
        """Label is plain English for investigators; the raw feature name is
        kept in `technical` so a reason code stays auditable to the model."""
        from app import make_reason_codes
        shap_row   = np.array([0.5, 0.1])
        feat_names = ["BasePolicy_Liability", "Age"]
        cat_cols   = ["BasePolicy"]
        codes = make_reason_codes(shap_row, feat_names, cat_cols, top_n=1)
        assert "base policy" in codes[0]["label"]
        assert "Liability" in codes[0]["label"]
        assert codes[0]["technical"] == "BasePolicy_Liability"

    def test_ohe_present_reads_as_is(self):
        from app import make_reason_codes
        codes = make_reason_codes(
            np.array([0.5, 0.1]), ["BasePolicy_Liability", "Age"], ["BasePolicy"],
            top_n=1, feature_values=np.array([1.0, 0.0]),
        )
        assert "is Liability" in codes[0]["label"]
        assert "not" not in codes[0]["label"]

    def test_ohe_absent_reads_as_is_not(self):
        """A dummy of 0 with a large SHAP value means the claim is NOT that
        category — asserting it IS contradicts the claim's own attributes."""
        from app import make_reason_codes
        codes = make_reason_codes(
            np.array([0.5, 0.1]), ["BasePolicy_Liability", "Age"], ["BasePolicy"],
            top_n=1, feature_values=np.array([0.0, 0.0]),
        )
        assert "is not Liability" in codes[0]["label"]

    def test_engineered_feature_gets_plain_english(self):
        from app import make_reason_codes
        codes = make_reason_codes(
            np.array([1.7]), ["AddressChange_Claim_Num"], [], top_n=1,
        )
        assert "address change" in codes[0]["label"].lower()
        assert "AddressChange_Claim_Num" == codes[0]["technical"]

    def test_unknown_feature_degrades_readably(self):
        """A feature with no label must not raise — it prettifies instead."""
        from app import make_reason_codes
        codes = make_reason_codes(np.array([0.4]), ["SomeNewFeature_x_Thing"], [], top_n=1)
        assert "some new feature" in codes[0]["label"]


# ============================================================
# Tests: SHAP reason code (shap_explainability.py)
# ============================================================
class TestMakeReasonCode:

    def test_high_risk_positive_shap(self):
        from shap_explainability import make_reason_code
        rc = make_reason_code("Age", 0.5)
        assert "High risk" in rc
        assert "Age" in rc

    def test_low_risk_negative_shap(self):
        from shap_explainability import make_reason_code
        rc = make_reason_code("Age", -0.5)
        assert "Low risk" in rc

    def test_strong_intensity(self):
        from shap_explainability import make_reason_code
        rc = make_reason_code("feature", 0.9)
        assert "STRONG" in rc

    def test_moderate_intensity(self):
        from shap_explainability import make_reason_code
        rc = make_reason_code("feature", 0.3)
        assert "MODERATE" in rc

    def test_mild_intensity(self):
        from shap_explainability import make_reason_code
        rc = make_reason_code("feature", 0.1)
        assert "MILD" in rc

    def test_ohe_decoding_with_underscore_in_column(self):
        from shap_explainability import make_reason_code
        # Column name contains underscore — longest prefix match must work
        rc = make_reason_code(
            "AddressChange_Claim_2 to 3 years",
            0.12,
            cat_cols=["AddressChange_Claim", "Claim"],
        )
        assert "AddressChange_Claim" in rc
        assert "2 to 3 years" in rc

    def test_plain_numeric_feature(self):
        from shap_explainability import make_reason_code
        rc = make_reason_code("Deductible", 0.08, cat_cols=["BasePolicy"])
        assert "Deductible" in rc
