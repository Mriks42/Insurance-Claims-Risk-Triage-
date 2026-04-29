"""
Tests for data_preprocessing.py and data_validation.py
========================================================
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def minimal_df():
    """Minimal valid dataframe that mimics fraud_oracle structure."""
    rows = []
    for i in range(200):
        rows.append({
            "PolicyNumber": i + 1,
            "WeekOfMonth": 2,
            "WeekOfMonthClaimed": 3,
            "Age": 30 + (i % 40),
            "RepNumber": 5,
            "Deductible": [300, 400, 500, 700][i % 4],
            "DriverRating": (i % 4) + 1,
            "Year": 2024,
            "Month": "Jan",
            "DayOfWeek": "Monday",
            "Make": "Honda",
            "AccidentArea": "Urban",
            "DayOfWeekClaimed": "Wednesday",
            "MonthClaimed": "Feb",
            "Sex": "Male" if i % 2 == 0 else "Female",
            "MaritalStatus": "Single",
            "Fault": "Policy Holder" if i % 3 == 0 else "Third Party",
            "PolicyType": "Sedan - Collision",
            "VehicleCategory": "Sedan",
            "VehiclePrice": "20000 to 29000",
            "Days_Policy_Accident": "more than 30",
            "Days_Policy_Claim": "more than 30",
            "PastNumberOfClaims": "none",
            "AgeOfVehicle": "3 years",
            "AgeOfPolicyHolder": "26 to 30",
            "PoliceReportFiled": "Yes",
            "WitnessPresent": "Yes",
            "AgentType": "Internal",
            "NumberOfSuppliments": "none",
            "AddressChange_Claim": "no change",
            "NumberOfCars": "1 vehicle",
            "BasePolicy": "Collision",
            "FraudFound_P": 1 if i < 12 else 0,  # ~6% fraud rate
        })
    return pd.DataFrame(rows)


# ============================================================
# Tests: data split
# ============================================================
class TestDataSplit:

    def test_split_sizes(self, minimal_df):
        from data_preprocessing import prepare_features, split_data
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        total = len(minimal_df)
        assert len(X_train) + len(X_val) + len(X_test) == total

    def test_train_is_80_percent(self, minimal_df):
        from data_preprocessing import prepare_features, split_data
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        train_pct = len(X_train) / len(minimal_df)
        assert abs(train_pct - 0.80) < 0.05

    def test_val_test_equal_size(self, minimal_df):
        from data_preprocessing import prepare_features, split_data
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        assert len(X_val) == len(X_test)

    def test_stratification_preserves_fraud_rate(self, minimal_df):
        from data_preprocessing import prepare_features, split_data
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        base_rate = y.mean()
        # Each split should be within 5% of the base fraud rate
        assert abs(y_train.mean() - base_rate) < 0.05
        assert abs(y_val.mean() - base_rate) < 0.05
        assert abs(y_test.mean() - base_rate) < 0.05

    def test_no_overlap_between_splits(self, minimal_df):
        from data_preprocessing import prepare_features, split_data
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        train_idx = set(X_train.index)
        val_idx   = set(X_val.index)
        test_idx  = set(X_test.index)

        assert len(train_idx & val_idx) == 0,  "Train/val overlap"
        assert len(train_idx & test_idx) == 0, "Train/test overlap"
        assert len(val_idx & test_idx) == 0,   "Val/test overlap"


# ============================================================
# Tests: preprocessor
# ============================================================
class TestPreprocessor:

    def test_preprocessor_fits_without_error(self, minimal_df):
        from data_preprocessing import prepare_features, split_data, build_preprocessor
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
        assert preprocessor is not None

    def test_transform_produces_numeric_output(self, minimal_df):
        from data_preprocessing import prepare_features, split_data, build_preprocessor, transform_data
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
        X_train_t, X_val_t, X_test_t = transform_data(preprocessor, X_train, X_val, X_test)

        assert X_train_t.dtype in [np.float32, np.float64]
        assert X_val_t.dtype in [np.float32, np.float64]
        assert X_test_t.dtype in [np.float32, np.float64]

    def test_transform_no_nan(self, minimal_df):
        from data_preprocessing import prepare_features, split_data, build_preprocessor, transform_data
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
        X_train_t, X_val_t, X_test_t = transform_data(preprocessor, X_train, X_val, X_test)

        assert not np.isnan(X_train_t).any(), "NaN in training transform"
        assert not np.isnan(X_val_t).any(),   "NaN in val transform"
        assert not np.isnan(X_test_t).any(),  "NaN in test transform"

    def test_feature_names_match_columns(self, minimal_df):
        from data_preprocessing import (
            prepare_features, split_data, build_preprocessor,
            transform_data, get_feature_names
        )
        X, y, _ = prepare_features(minimal_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        preprocessor, cat_cols, num_cols = build_preprocessor(X_train)
        X_train_t, _, _ = transform_data(preprocessor, X_train, X_val, X_test)
        feature_names = get_feature_names(preprocessor, num_cols, cat_cols)

        assert len(feature_names) == X_train_t.shape[1]

    def test_policy_number_dropped(self, minimal_df):
        from data_preprocessing import prepare_features
        X, y, _ = prepare_features(minimal_df)
        assert "PolicyNumber" not in X.columns


# ============================================================
# Tests: data validation
# ============================================================
class TestDataValidation:

    def test_valid_data_passes(self, minimal_df):
        from data_validation import validate_and_report
        report = validate_and_report(minimal_df)
        assert report["all_passed"] is True

    def test_report_has_required_keys(self, minimal_df):
        from data_validation import validate_and_report
        report = validate_and_report(minimal_df)
        for key in ["total_rows", "total_cols", "fraud_rate",
                    "missing_values", "duplicate_rows", "checks", "all_passed"]:
            assert key in report

    def test_fraud_rate_correct(self, minimal_df):
        from data_validation import validate_and_report
        report = validate_and_report(minimal_df)
        expected = minimal_df["FraudFound_P"].mean()
        assert abs(report["fraud_rate"] - expected) < 0.001

    def test_missing_values_detected(self, minimal_df):
        from data_validation import validate_and_report
        df_with_nulls = minimal_df.copy()
        df_with_nulls.loc[0, "Age"] = np.nan
        report = validate_and_report(df_with_nulls)
        assert report["missing_values"] > 0

    def test_duplicate_rows_detected(self, minimal_df):
        from data_validation import validate_and_report
        df_with_dups = pd.concat([minimal_df, minimal_df.iloc[:5]], ignore_index=True)
        report = validate_and_report(df_with_dups)
        assert report["duplicate_rows"] > 0
