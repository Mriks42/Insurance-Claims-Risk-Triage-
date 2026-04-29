"""
Tests for feature_engineering.py
==================================
Verifies that engineer_features() produces the correct columns,
correct values, and handles edge cases properly.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import (
    engineer_features,
    get_engineered_numeric_cols,
    REPLACED_CATEGORICALS,
    DAYS_POLICY_MAP,
    PAST_CLAIMS_MAP,
    AGE_VEHICLE_MAP,
    VEHICLE_PRICE_MAP,
)


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def sample_row():
    """A single realistic claim row."""
    return {
        "PolicyNumber": 12345,
        "WeekOfMonth": 2,
        "WeekOfMonthClaimed": 3,
        "Age": 23,
        "RepNumber": 5,
        "Deductible": 400,
        "DriverRating": 3,
        "Year": 2024,
        "Month": "Jan",
        "DayOfWeek": "Saturday",
        "Make": "Honda",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Monday",
        "MonthClaimed": "Feb",
        "Sex": "Male",
        "MaritalStatus": "Single",
        "Fault": "Policy Holder",
        "PolicyType": "Sport - Liability",
        "VehicleCategory": "Sport",
        "VehiclePrice": "more than 69000",
        "Days_Policy_Accident": "none",
        "Days_Policy_Claim": "1 to 7",
        "PastNumberOfClaims": "1",
        "AgeOfVehicle": "3 years",
        "AgeOfPolicyHolder": "21 to 25",
        "PoliceReportFiled": "No",
        "WitnessPresent": "No",
        "AgentType": "External",
        "NumberOfSuppliments": "3 to 5",
        "AddressChange_Claim": "under 6 months",
        "NumberOfCars": "1 vehicle",
        "BasePolicy": "Liability",
        "FraudFound_P": 1,
    }


@pytest.fixture
def sample_df(sample_row):
    """DataFrame with 5 identical rows for batch testing."""
    return pd.DataFrame([sample_row] * 5)


@pytest.fixture
def low_risk_row():
    """A low-risk claim row."""
    return {
        "PolicyNumber": 99999,
        "WeekOfMonth": 1,
        "WeekOfMonthClaimed": 2,
        "Age": 42,
        "RepNumber": 3,
        "Deductible": 300,
        "DriverRating": 1,
        "Year": 2024,
        "Month": "Mar",
        "DayOfWeek": "Monday",
        "Make": "Toyota",
        "AccidentArea": "Urban",
        "DayOfWeekClaimed": "Wednesday",
        "MonthClaimed": "Mar",
        "Sex": "Female",
        "MaritalStatus": "Married",
        "Fault": "Third Party",
        "PolicyType": "Sedan - Collision",
        "VehicleCategory": "Sedan",
        "VehiclePrice": "20000 to 29000",
        "Days_Policy_Accident": "more than 30",
        "Days_Policy_Claim": "more than 30",
        "PastNumberOfClaims": "none",
        "AgeOfVehicle": "4 years",
        "AgeOfPolicyHolder": "41 to 50",
        "PoliceReportFiled": "Yes",
        "WitnessPresent": "Yes",
        "AgentType": "Internal",
        "NumberOfSuppliments": "none",
        "AddressChange_Claim": "no change",
        "NumberOfCars": "2 vehicles",
        "BasePolicy": "Collision",
        "FraudFound_P": 0,
    }


# ============================================================
# Tests: output shape and columns
# ============================================================
class TestOutputShape:

    def test_returns_dataframe(self, sample_df):
        result = engineer_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_original_columns_preserved(self, sample_df):
        result = engineer_features(sample_df)
        for col in sample_df.columns:
            assert col in result.columns, f"Original column '{col}' missing from output"

    def test_row_count_unchanged(self, sample_df):
        result = engineer_features(sample_df)
        assert len(result) == len(sample_df)

    def test_new_columns_added(self, sample_df):
        result = engineer_features(sample_df)
        new_cols = [c for c in result.columns if c not in sample_df.columns]
        assert len(new_cols) > 0, "No new columns were added"

    def test_engineered_numeric_cols_present(self, sample_df):
        result = engineer_features(sample_df)
        expected = get_engineered_numeric_cols()
        for col in expected:
            assert col in result.columns, f"Expected engineered column '{col}' missing"

    def test_does_not_modify_original(self, sample_df):
        original_cols = list(sample_df.columns)
        original_len  = len(sample_df)
        _ = engineer_features(sample_df)
        assert list(sample_df.columns) == original_cols
        assert len(sample_df) == original_len


# ============================================================
# Tests: ordinal encoding correctness
# ============================================================
class TestOrdinalEncoding:

    def test_days_policy_accident_none(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["Days_Policy_Accident_Num"] == 0).all()

    def test_days_policy_claim_1_to_7(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["Days_Policy_Claim_Num"] == 4).all()

    def test_vehicle_price_mapping(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["VehiclePrice_Num"] == 80000).all()

    def test_past_claims_mapping(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["PastNumberOfClaims_Num"] == 1).all()

    def test_age_vehicle_mapping(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["AgeOfVehicle_Num"] == 3).all()

    def test_low_risk_no_past_claims(self):
        df = pd.DataFrame([{
            "PolicyNumber": 1, "WeekOfMonth": 1, "WeekOfMonthClaimed": 1,
            "Age": 40, "RepNumber": 1, "Deductible": 300, "DriverRating": 1,
            "Year": 2024, "Month": "Jan", "DayOfWeek": "Monday", "Make": "Toyota",
            "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan",
            "Sex": "Male", "MaritalStatus": "Married", "Fault": "Third Party",
            "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
            "VehiclePrice": "20000 to 29000", "Days_Policy_Accident": "more than 30",
            "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
            "AgeOfVehicle": "new", "AgeOfPolicyHolder": "36 to 40",
            "PoliceReportFiled": "Yes", "WitnessPresent": "Yes", "AgentType": "Internal",
            "NumberOfSuppliments": "none", "AddressChange_Claim": "no change",
            "NumberOfCars": "1 vehicle", "BasePolicy": "Collision", "FraudFound_P": 0,
        }])
        result = engineer_features(df)
        assert result["PastNumberOfClaims_Num"].iloc[0] == 0
        assert result["AgeOfVehicle_Num"].iloc[0] == 0


# ============================================================
# Tests: binary risk flags
# ============================================================
class TestRiskFlags:

    def test_no_police_report_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["NoPoliceReport"] == 1).all()

    def test_no_witness_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["NoWitness"] == 1).all()

    def test_no_report_no_witness_combined(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["NoReportNoWitness"] == 1).all()

    def test_policy_holder_fault_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["PolicyHolderFault"] == 1).all()

    def test_external_agent_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["ExternalAgent"] == 1).all()

    def test_young_driver_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["YoungDriver"] == 1).all()

    def test_not_young_driver(self):
        df = pd.DataFrame([{
            "PolicyNumber": 1, "WeekOfMonth": 1, "WeekOfMonthClaimed": 1,
            "Age": 40, "RepNumber": 1, "Deductible": 300, "DriverRating": 1,
            "Year": 2024, "Month": "Jan", "DayOfWeek": "Monday", "Make": "Toyota",
            "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan",
            "Sex": "Male", "MaritalStatus": "Married", "Fault": "Third Party",
            "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
            "VehiclePrice": "20000 to 29000", "Days_Policy_Accident": "more than 30",
            "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
            "AgeOfVehicle": "4 years", "AgeOfPolicyHolder": "36 to 40",
            "PoliceReportFiled": "Yes", "WitnessPresent": "Yes", "AgentType": "Internal",
            "NumberOfSuppliments": "none", "AddressChange_Claim": "no change",
            "NumberOfCars": "1 vehicle", "BasePolicy": "Collision", "FraudFound_P": 0,
        }])
        result = engineer_features(df)
        assert result["YoungDriver"].iloc[0] == 0

    def test_weekend_accident_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["WeekendAccident"] == 1).all()

    def test_weekday_not_weekend(self):
        df = pd.DataFrame([{
            "PolicyNumber": 1, "WeekOfMonth": 1, "WeekOfMonthClaimed": 1,
            "Age": 30, "RepNumber": 1, "Deductible": 300, "DriverRating": 1,
            "Year": 2024, "Month": "Jan", "DayOfWeek": "Monday", "Make": "Toyota",
            "AccidentArea": "Urban", "DayOfWeekClaimed": "Tuesday", "MonthClaimed": "Jan",
            "Sex": "Male", "MaritalStatus": "Married", "Fault": "Third Party",
            "PolicyType": "Sedan - Collision", "VehicleCategory": "Sedan",
            "VehiclePrice": "20000 to 29000", "Days_Policy_Accident": "more than 30",
            "Days_Policy_Claim": "more than 30", "PastNumberOfClaims": "none",
            "AgeOfVehicle": "4 years", "AgeOfPolicyHolder": "26 to 30",
            "PoliceReportFiled": "Yes", "WitnessPresent": "Yes", "AgentType": "Internal",
            "NumberOfSuppliments": "none", "AddressChange_Claim": "no change",
            "NumberOfCars": "1 vehicle", "BasePolicy": "Collision", "FraudFound_P": 0,
        }])
        result = engineer_features(df)
        assert result["WeekendAccident"].iloc[0] == 0


# ============================================================
# Tests: guideline-grounded flags
# ============================================================
class TestGuidelineFlags:

    def test_liability_no_police(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["Liability_NoPolice"] == 1).all()

    def test_very_early_claim_flag(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["VeryEarlyClaimFlag"] == 1).all()

    def test_sport_young_driver(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["Sport_YoungDriver"] == 1).all()

    def test_external_agent_no_police(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["ExternalAgent_NoPolice"] == 1).all()

    def test_low_risk_no_guideline_flags(self, low_risk_row):
        df = pd.DataFrame([low_risk_row])
        result = engineer_features(df)
        assert result["Liability_NoPolice"].iloc[0] == 0
        assert result["VeryEarlyClaimFlag"].iloc[0] == 0
        assert result["Sport_YoungDriver"].iloc[0] == 0
        assert result["ExternalAgent_NoPolice"].iloc[0] == 0


# ============================================================
# Tests: risk flag count
# ============================================================
class TestRiskFlagCount:

    def test_risk_flag_count_is_non_negative(self, sample_df):
        result = engineer_features(sample_df)
        assert (result["RiskFlagCount"] >= 0).all()

    def test_risk_flag_count_high_risk(self, sample_df):
        result = engineer_features(sample_df)
        # High-risk claim should have multiple flags
        assert result["RiskFlagCount"].iloc[0] >= 5

    def test_risk_flag_count_low_risk(self, low_risk_row):
        df = pd.DataFrame([low_risk_row])
        result = engineer_features(df)
        assert result["RiskFlagCount"].iloc[0] < 3


# ============================================================
# Tests: interaction features
# ============================================================
class TestInteractionFeatures:

    def test_age_x_deductible(self, sample_df):
        result = engineer_features(sample_df)
        expected = 23 * 400
        assert (result["Age_x_Deductible"] == expected).all()

    def test_fault_no_police_interaction(self, sample_df):
        result = engineer_features(sample_df)
        # PolicyHolderFault=1, NoPoliceReport=1 → 1
        assert (result["Fault_NoPolice"] == 1).all()

    def test_price_to_deductible_ratio(self, sample_df):
        result = engineer_features(sample_df)
        expected = 80000 / (400 + 1)
        assert abs(result["PriceToDeductible"].iloc[0] - expected) < 0.01


# ============================================================
# Tests: REPLACED_CATEGORICALS list
# ============================================================
class TestReplacedCategoricals:

    def test_replaced_categoricals_are_in_raw_data(self, sample_df):
        for col in REPLACED_CATEGORICALS:
            assert col in sample_df.columns, f"'{col}' not in sample data"

    def test_replaced_categoricals_have_numeric_versions(self, sample_df):
        result = engineer_features(sample_df)
        # Spot check a few
        assert "Days_Policy_Accident_Num" in result.columns
        assert "VehiclePrice_Num" in result.columns
        assert "PastNumberOfClaims_Num" in result.columns
