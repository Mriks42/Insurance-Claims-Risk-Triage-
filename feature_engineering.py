"""
Feature Engineering Module
===========================
Creates new features from the raw fraud_oracle dataset to improve
model performance. These transformations happen BEFORE the sklearn
preprocessor (which handles imputation + scaling + one-hot encoding).

Key ideas:
1. Extract numeric values from categorical-but-ordinal columns
2. Create interaction / ratio features
3. Add time-based and risk-indicator flags
4. Provide both XGBoost-style (one-hot) and CatBoost-style (native cat) outputs

Usage:
    from feature_engineering import engineer_features
"""

import pandas as pd
import numpy as np


# ============================================================
# Ordinal mappings: convert string categories → numeric
# ============================================================
DAYS_POLICY_MAP = {
    "none": 0,
    "1 to 7": 4,
    "8 to 15": 12,
    "15 to 30": 22,
    "more than 30": 45,
}

PAST_CLAIMS_MAP = {
    "none": 0,
    "1": 1,
    "2 to 4": 3,
    "more than 4": 6,
}

AGE_VEHICLE_MAP = {
    "new": 0,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "more than 7": 10,
}

AGE_POLICY_HOLDER_MAP = {
    "16 to 17": 16.5,
    "18 to 20": 19,
    "21 to 25": 23,
    "26 to 30": 28,
    "31 to 35": 33,
    "36 to 40": 38,
    "41 to 50": 45.5,
    "51 to 65": 58,
    "over 65": 70,
}

NUM_SUPPLIMENTS_MAP = {
    "none": 0,
    "1 to 2": 1.5,
    "3 to 5": 4,
    "more than 5": 7,
}

NUM_CARS_MAP = {
    "1 vehicle": 1,
    "2 vehicles": 2,
    "3 to 4": 3.5,
    "5 to 8": 6.5,
    "more than 8": 10,
}

ADDRESS_CHANGE_MAP = {
    "no change": 0,
    "under 6 months": 0.25,
    "1 year": 1,
    "2 to 3 years": 2.5,
    "4 to 8 years": 6,
}

VEHICLE_PRICE_MAP = {
    "less than 20000": 15000,
    "20000 to 29000": 25000,
    "30000 to 39000": 35000,
    "40000 to 59000": 50000,
    "60000 to 69000": 65000,
    "more than 69000": 80000,
}

# Month → numeric
MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Day of week → numeric (0=Monday)
DOW_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}


# ============================================================
# Main feature engineering function
# ============================================================
def engineer_features(df):
    """
    Engineer new features from the raw dataframe.
    Returns a new dataframe with all original + engineered columns.
    Does NOT drop any columns (that's handled by the preprocessor).
    """
    df = df.copy()

    # ----------------------------------------------------------
    # 1) Convert ordinal categoricals to numeric
    # ----------------------------------------------------------
    df["Days_Policy_Accident_Num"] = df["Days_Policy_Accident"].map(DAYS_POLICY_MAP)
    df["Days_Policy_Claim_Num"] = df["Days_Policy_Claim"].map(DAYS_POLICY_MAP)
    df["PastNumberOfClaims_Num"] = df["PastNumberOfClaims"].map(PAST_CLAIMS_MAP)
    df["AgeOfVehicle_Num"] = df["AgeOfVehicle"].map(AGE_VEHICLE_MAP)
    df["AgeOfPolicyHolder_Num"] = df["AgeOfPolicyHolder"].map(AGE_POLICY_HOLDER_MAP)
    df["NumberOfSuppliments_Num"] = df["NumberOfSuppliments"].map(NUM_SUPPLIMENTS_MAP)
    df["NumberOfCars_Num"] = df["NumberOfCars"].map(NUM_CARS_MAP)
    df["AddressChange_Claim_Num"] = df["AddressChange_Claim"].map(ADDRESS_CHANGE_MAP)
    df["VehiclePrice_Num"] = df["VehiclePrice"].map(VEHICLE_PRICE_MAP)

    # Month and day of week as numbers
    df["Month_Num"] = df["Month"].map(MONTH_MAP)
    df["MonthClaimed_Num"] = df["MonthClaimed"].map(MONTH_MAP)
    df["DayOfWeek_Num"] = df["DayOfWeek"].map(DOW_MAP)
    df["DayOfWeekClaimed_Num"] = df["DayOfWeekClaimed"].map(DOW_MAP)

    # ----------------------------------------------------------
    # 2) Time-based features
    # ----------------------------------------------------------
    # Gap between week of month (accident vs claim)
    df["WeekGap"] = (df["WeekOfMonthClaimed"] - df["WeekOfMonth"]).abs()

    # Month gap between accident and claim (circular)
    df["MonthGap"] = df.apply(
        lambda r: _month_gap(r.get("Month_Num"), r.get("MonthClaimed_Num")),
        axis=1,
    )

    # Same day-of-week for accident and claim?
    df["SameDayOfWeek"] = (df["DayOfWeek"] == df["DayOfWeekClaimed"]).astype(int)

    # ----------------------------------------------------------
    # 3) Binary risk indicators
    # ----------------------------------------------------------
    # Weekend accident
    df["WeekendAccident"] = df["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)
    # Weekend claim
    df["WeekendClaim"] = df["DayOfWeekClaimed"].isin(["Saturday", "Sunday"]).astype(int)

    # No police report filed (risk flag)
    df["NoPoliceReport"] = (df["PoliceReportFiled"] == "No").astype(int)
    # No witness present (risk flag)
    df["NoWitness"] = (df["WitnessPresent"] == "No").astype(int)
    # Both no police report AND no witness
    df["NoReportNoWitness"] = (df["NoPoliceReport"] & df["NoWitness"]).astype(int)

    # Policy holder at fault
    df["PolicyHolderFault"] = (df["Fault"] == "Policy Holder").astype(int)

    # External agent
    df["ExternalAgent"] = (df["AgentType"] == "External").astype(int)

    # High deductible (top quartile)
    df["HighDeductible"] = (df["Deductible"] >= df["Deductible"].quantile(0.75)).astype(int)

    # Young driver
    df["YoungDriver"] = (df["Age"] <= 25).astype(int)

    # Has past claims
    df["HasPastClaims"] = (df["PastNumberOfClaims"] != "none").astype(int)

    # Recent address change
    df["RecentAddressChange"] = df["AddressChange_Claim"].isin(
        ["under 6 months", "1 year"]
    ).astype(int)

    # ----------------------------------------------------------
    # 4) Interaction / ratio features
    # ----------------------------------------------------------
    # Age × Deductible interaction
    df["Age_x_Deductible"] = df["Age"] * df["Deductible"]

    # Fault + No police report interaction
    df["Fault_NoPolice"] = df["PolicyHolderFault"] * df["NoPoliceReport"]

    # Vehicle price to deductible ratio
    df["PriceToDeductible"] = df["VehiclePrice_Num"] / (df["Deductible"] + 1)

    # Age × past claims
    df["Age_x_PastClaims"] = df["Age"] * df["PastNumberOfClaims_Num"]

    # Number of cars × number of supplements
    df["Cars_x_Suppliments"] = df["NumberOfCars_Num"] * df["NumberOfSuppliments_Num"]

    # Policy age (days since policy)
    df["PolicyAge_x_ClaimDelay"] = df["Days_Policy_Accident_Num"] * df["Days_Policy_Claim_Num"]

    # ----------------------------------------------------------
    # 5) Count-based features
    # ----------------------------------------------------------
    # Total risk flags
    df["RiskFlagCount"] = (
        df["NoPoliceReport"] +
        df["NoWitness"] +
        df["PolicyHolderFault"] +
        df["ExternalAgent"] +
        df["HighDeductible"] +
        df["YoungDriver"] +
        df["HasPastClaims"] +
        df["RecentAddressChange"]
    )

    return df


def _month_gap(m1, m2):
    """Circular month gap (0-6)."""
    if pd.isna(m1) or pd.isna(m2):
        return np.nan
    diff = abs(int(m1) - int(m2))
    return min(diff, 12 - diff)


# ============================================================
# Get the list of new numeric columns we added
# ============================================================
def get_engineered_numeric_cols():
    """Return names of all engineered numeric columns."""
    return [
        "Days_Policy_Accident_Num", "Days_Policy_Claim_Num",
        "PastNumberOfClaims_Num", "AgeOfVehicle_Num",
        "AgeOfPolicyHolder_Num", "NumberOfSuppliments_Num",
        "NumberOfCars_Num", "AddressChange_Claim_Num", "VehiclePrice_Num",
        "Month_Num", "MonthClaimed_Num", "DayOfWeek_Num", "DayOfWeekClaimed_Num",
        "WeekGap", "MonthGap", "SameDayOfWeek",
        "WeekendAccident", "WeekendClaim",
        "NoPoliceReport", "NoWitness", "NoReportNoWitness",
        "PolicyHolderFault", "ExternalAgent", "HighDeductible",
        "YoungDriver", "HasPastClaims", "RecentAddressChange",
        "Age_x_Deductible", "Fault_NoPolice", "PriceToDeductible",
        "Age_x_PastClaims", "Cars_x_Suppliments", "PolicyAge_x_ClaimDelay",
        "RiskFlagCount",
    ]


# ============================================================
# Columns to drop AFTER engineering (originals replaced by numeric)
# These are the categorical columns that we encoded to numeric above.
# We keep them for CatBoost (which handles cats natively) but
# drop them for XGBoost (to avoid redundancy with the numeric versions).
# ============================================================
REPLACED_CATEGORICALS = [
    "Days_Policy_Accident", "Days_Policy_Claim",
    "PastNumberOfClaims", "AgeOfVehicle", "AgeOfPolicyHolder",
    "NumberOfSuppliments", "NumberOfCars", "AddressChange_Claim",
    "VehiclePrice",
    "Month", "MonthClaimed", "DayOfWeek", "DayOfWeekClaimed",
    "PoliceReportFiled", "WitnessPresent", "AgentType", "Fault",
]


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("fraud_oracle.csv")
    df_eng = engineer_features(df)
    print(f"Original columns: {len(df.columns)}")
    print(f"After engineering: {len(df_eng.columns)}")
    new_cols = [c for c in df_eng.columns if c not in df.columns]
    print(f"New columns ({len(new_cols)}):")
    for c in new_cols:
        print(f"  {c}: {df_eng[c].dtype}, nunique={df_eng[c].nunique()}, "
              f"nulls={df_eng[c].isna().sum()}")
