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

def derive_age_band(age):
    """
    Map a numeric age onto the AgeOfPolicyHolder band the raw dataset uses.

    Needed by the live-scoring form, which collects Age as a slider but must
    also supply AgeOfPolicyHolder. Hardcoding a band there meant an age-70 claim
    was scored with a policyholder band of "26 to 30" — two contradictory
    signals fed to the model for the same person.

    Returned strings are exactly the keys of AGE_POLICY_HOLDER_MAP.
    """
    if age <= 17:
        return "16 to 17"
    if age <= 20:
        return "18 to 20"
    if age <= 25:
        return "21 to 25"
    if age <= 30:
        return "26 to 30"
    if age <= 35:
        return "31 to 35"
    if age <= 40:
        return "36 to 40"
    if age <= 50:
        return "41 to 50"
    if age <= 65:
        return "51 to 65"
    return "over 65"


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

# Deductible value at/above which a claim counts as "high deductible".
# Constant rather than a data-derived quantile — see engineer_features().
HIGH_DEDUCTIBLE_THRESHOLD = 400

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

    Implementation note — why the columns are collected in a dict and attached
    with a single concat rather than assigned one at a time:

    Profiling the inference API showed engineer_features accounting for ~72% of
    single-claim latency, and inside it the cost was 41 separate `df[col] = ...`
    assignments. Each one triggers a pandas block-manager insert, so the price is
    paid per column regardless of how little arithmetic is involved. On a 1-row
    request frame that is pure overhead. Building a plain dict and concatenating
    once turns 41 inserts into one, and is what took p50 single-claim scoring
    from ~43 ms to single digits. Values and column order are unchanged — there
    is a regression test asserting frame equality against the previous output.
    """
    df = df.copy()
    new = {}                       # column name -> Series/array, attached at the end

    # ----------------------------------------------------------
    # 1) Convert ordinal categoricals to numeric
    # ----------------------------------------------------------
    new["Days_Policy_Accident_Num"] = df["Days_Policy_Accident"].map(DAYS_POLICY_MAP)
    new["Days_Policy_Claim_Num"] = df["Days_Policy_Claim"].map(DAYS_POLICY_MAP)
    new["PastNumberOfClaims_Num"] = df["PastNumberOfClaims"].map(PAST_CLAIMS_MAP)
    new["AgeOfVehicle_Num"] = df["AgeOfVehicle"].map(AGE_VEHICLE_MAP)
    new["AgeOfPolicyHolder_Num"] = df["AgeOfPolicyHolder"].map(AGE_POLICY_HOLDER_MAP)
    new["NumberOfSuppliments_Num"] = df["NumberOfSuppliments"].map(NUM_SUPPLIMENTS_MAP)
    new["NumberOfCars_Num"] = df["NumberOfCars"].map(NUM_CARS_MAP)
    new["AddressChange_Claim_Num"] = df["AddressChange_Claim"].map(ADDRESS_CHANGE_MAP)
    new["VehiclePrice_Num"] = df["VehiclePrice"].map(VEHICLE_PRICE_MAP)

    # Month and day of week as numbers
    new["Month_Num"] = df["Month"].map(MONTH_MAP)
    new["MonthClaimed_Num"] = df["MonthClaimed"].map(MONTH_MAP)
    new["DayOfWeek_Num"] = df["DayOfWeek"].map(DOW_MAP)
    new["DayOfWeekClaimed_Num"] = df["DayOfWeekClaimed"].map(DOW_MAP)

    # ----------------------------------------------------------
    # 2) Time-based features
    # ----------------------------------------------------------
    # Gap between week of month (accident vs claim)
    new["WeekGap"] = (df["WeekOfMonthClaimed"] - df["WeekOfMonth"]).abs()

    # Month gap between accident and claim (circular). Vectorised: the previous
    # row-wise df.apply cost more than every other feature combined on batches.
    m1, m2 = new["Month_Num"], new["MonthClaimed_Num"]
    diff = (m1 - m2).abs()
    new["MonthGap"] = np.minimum(diff, 12 - diff).where(m1.notna() & m2.notna())

    # Same day-of-week for accident and claim?
    new["SameDayOfWeek"] = (df["DayOfWeek"] == df["DayOfWeekClaimed"]).astype(int)

    # ----------------------------------------------------------
    # 3) Binary risk indicators
    # ----------------------------------------------------------
    new["WeekendAccident"] = df["DayOfWeek"].isin(["Saturday", "Sunday"]).astype(int)
    new["WeekendClaim"] = df["DayOfWeekClaimed"].isin(["Saturday", "Sunday"]).astype(int)

    new["NoPoliceReport"] = (df["PoliceReportFiled"] == "No").astype(int)
    new["NoWitness"] = (df["WitnessPresent"] == "No").astype(int)
    new["NoReportNoWitness"] = (new["NoPoliceReport"] & new["NoWitness"]).astype(int)

    new["PolicyHolderFault"] = (df["Fault"] == "Policy Holder").astype(int)
    new["ExternalAgent"] = (df["AgentType"] == "External").astype(int)

    # High deductible.
    # This threshold used to be computed as df["Deductible"].quantile(0.75) over
    # whatever frame was passed in — i.e. fitted on the full dataset including the
    # test rows, and different for a single-claim frame in the live-scoring path.
    # It is pinned to the constant that expression produced on the full training
    # data (400) so the feature is split-independent and identical for one claim
    # or 15,420 of them.
    # NOTE: Deductible is 400 for 14,838 of 15,420 rows, so this flag is ~1 for
    # 99.9% of claims and carries almost no signal. Raising it to >= 500 (3.7% of
    # claims) is the sensible fix, but that changes the feature space and requires
    # retraining, so it is left for the next training run.
    new["HighDeductible"] = (df["Deductible"] >= HIGH_DEDUCTIBLE_THRESHOLD).astype(int)

    new["YoungDriver"] = (df["Age"] <= 25).astype(int)
    new["HasPastClaims"] = (df["PastNumberOfClaims"] != "none").astype(int)
    new["RecentAddressChange"] = df["AddressChange_Claim"].isin(
        ["under 6 months", "1 year"]
    ).astype(int)

    # ----------------------------------------------------------
    # 4) Interaction / ratio features
    # ----------------------------------------------------------
    new["Age_x_Deductible"] = df["Age"] * df["Deductible"]
    new["Fault_NoPolice"] = new["PolicyHolderFault"] * new["NoPoliceReport"]
    new["PriceToDeductible"] = new["VehiclePrice_Num"] / (df["Deductible"] + 1)
    new["Age_x_PastClaims"] = df["Age"] * new["PastNumberOfClaims_Num"]
    new["Cars_x_Suppliments"] = new["NumberOfCars_Num"] * new["NumberOfSuppliments_Num"]
    new["PolicyAge_x_ClaimDelay"] = (new["Days_Policy_Accident_Num"]
                                     * new["Days_Policy_Claim_Num"])

    # ----------------------------------------------------------
    # 5) Guideline-grounded combination flags
    #    (directly derived from fraud_red_flags.md and
    #     policy_coverage_standards.md)
    # ----------------------------------------------------------

    # Liability policy + no police report — highest-risk config per guidelines
    new["Liability_NoPolice"] = (
        (df["BasePolicy"] == "Liability") & (df["PoliceReportFiled"] == "No")
    ).astype(int)

    # High-value vehicle (>$60k) with liability-only coverage
    new["HighValue_Liability"] = (
        (new["VehiclePrice_Num"] >= 60000) & (df["BasePolicy"] == "Liability")
    ).astype(int)

    # Sport vehicle + young driver — explicitly flagged in guidelines
    new["Sport_YoungDriver"] = (
        (df["VehicleCategory"] == "Sport") & (new["YoungDriver"] == 1)
    ).astype(int)

    # Very early claim: accident within first 7 days of policy inception
    # "none" in Days_Policy_Accident means < 7 days — mandatory SIU per guidelines
    new["VeryEarlyClaimFlag"] = (df["Days_Policy_Accident"] == "none").astype(int)
    new["EarlyClaimFlag"] = (
        df["Days_Policy_Accident"].isin(["none", "1 to 7"])
    ).astype(int)

    # High supplement count (3+) — cost inflation indicator per guidelines
    new["HighSupplements"] = (new["NumberOfSuppliments_Num"] >= 4).astype(int)

    # External agent + no police report — historically higher fraud rate
    new["ExternalAgent_NoPolice"] = (
        (new["ExternalAgent"] == 1) & (new["NoPoliceReport"] == 1)
    ).astype(int)

    # ----------------------------------------------------------
    # 6) Count-based features
    # ----------------------------------------------------------
    # Total risk flags (expanded with new guideline flags)
    new["RiskFlagCount"] = (
        new["NoPoliceReport"] +
        new["NoWitness"] +
        new["PolicyHolderFault"] +
        new["ExternalAgent"] +
        new["HighDeductible"] +
        new["YoungDriver"] +
        new["HasPastClaims"] +
        new["RecentAddressChange"] +
        new["Liability_NoPolice"] +
        new["HighValue_Liability"] +
        new["Sport_YoungDriver"] +
        new["VeryEarlyClaimFlag"] +
        new["HighSupplements"] +
        new["ExternalAgent_NoPolice"]
    )

    # One insert instead of 41
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


# ============================================================
# Plain-English feature labels
# ============================================================
# Reason codes are read by claims investigators, not by the people who wrote
# the feature engineering. "Age_x_Deductible" and "AddressChange_Claim_Num" are
# internal names; these are what they mean. Anything missing falls back to a
# de-camel-cased version of the raw name, so a new feature degrades to readable
# rather than to a KeyError.
FEATURE_LABELS = {
    # ── ordinal encodings ──────────────────────────────────
    "Days_Policy_Accident_Num": "days from policy start to accident",
    "Days_Policy_Claim_Num":    "days from policy start to claim",
    "PastNumberOfClaims_Num":   "number of past claims",
    "AgeOfVehicle_Num":         "vehicle age",
    "AgeOfPolicyHolder_Num":    "policyholder age band",
    "NumberOfSuppliments_Num":  "number of supplements claimed",
    "NumberOfCars_Num":         "number of vehicles on the policy",
    "AddressChange_Claim_Num":  "time since the last address change",
    "VehiclePrice_Num":         "vehicle price",
    "Month_Num":                "month of accident",
    "MonthClaimed_Num":         "month claim was filed",
    "DayOfWeek_Num":            "day of week of accident",
    "DayOfWeekClaimed_Num":     "day of week claim was filed",
    # ── timing ─────────────────────────────────────────────
    "WeekGap":                  "weeks between accident and claim",
    "MonthGap":                 "months between accident and claim",
    "SameDayOfWeek":            "accident and claim fell on the same weekday",
    "WeekendAccident":          "accident happened at the weekend",
    "WeekendClaim":             "claim filed at the weekend",
    # ── risk indicators ────────────────────────────────────
    "NoPoliceReport":           "no police report filed",
    "NoWitness":                "no witness present",
    "NoReportNoWitness":        "neither a police report nor a witness",
    "PolicyHolderFault":        "policyholder was at fault",
    "ExternalAgent":            "claim handled by an external agent",
    "HighDeductible":           "high deductible",
    "YoungDriver":              "driver aged 25 or under",
    "HasPastClaims":            "has prior claims",
    "RecentAddressChange":      "address changed within the past year",
    # ── interactions ───────────────────────────────────────
    "Age_x_Deductible":         "age combined with deductible",
    "Fault_NoPolice":           "at fault with no police report",
    "PriceToDeductible":        "vehicle price relative to deductible",
    "Age_x_PastClaims":         "age combined with prior claim count",
    "Cars_x_Suppliments":       "vehicle count combined with supplements",
    "PolicyAge_x_ClaimDelay":   "policy age combined with claim delay",
    # ── guideline-grounded flags ───────────────────────────
    "Liability_NoPolice":       "liability-only cover with no police report",
    "HighValue_Liability":      "high-value vehicle on liability-only cover",
    "Sport_YoungDriver":        "sports vehicle with a driver aged 25 or under",
    "VeryEarlyClaimFlag":       "accident within days of the policy starting",
    "EarlyClaimFlag":           "accident within the first week of cover",
    "HighSupplements":          "three or more supplements claimed",
    "ExternalAgent_NoPolice":   "external agent with no police report",
    "RiskFlagCount":            "total number of risk flags raised",
    # ── raw columns kept by the model ──────────────────────
    "Age":                      "policyholder age",
    "Deductible":               "deductible amount",
    "DriverRating":             "driver rating",
    "RepNumber":                "claims representative",
    "Year":                     "policy year",
    "WeekOfMonth":              "week of month of accident",
    "WeekOfMonthClaimed":       "week of month claim was filed",
    # ── categorical columns (one-hot prefixes) ─────────────
    "BasePolicy":               "base policy",
    "PolicyType":               "policy type",
    "VehicleCategory":          "vehicle category",
    "Make":                     "vehicle make",
    "AccidentArea":             "accident area",
    "Sex":                      "sex",
    "MaritalStatus":            "marital status",
}


def _prettify(name):
    """Fallback label: split underscores and camelCase into readable words."""
    import re
    text = name.replace("_x_", " x ").replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return text.lower().strip()


def humanize_feature(feature_name, cat_cols=None, feature_value=None):
    """
    Turn a model feature name into something a claims investigator can read.

    One-hot columns are resolved against the claim's own value, so a dummy that
    is 0 reads "base policy is not Liability" rather than asserting the claim is
    Liability. Pass feature_value=None to keep the older neutral phrasing.
    """
    if cat_cols:
        for col in sorted(cat_cols, key=len, reverse=True):
            prefix = col + "_"
            if feature_name.startswith(prefix):
                level = feature_name[len(prefix):]
                label = FEATURE_LABELS.get(col, _prettify(col))
                if feature_value is None:
                    return f"{label} = '{level}'"
                if float(feature_value) > 0.5:
                    return f"{label} is {level}"
                return f"{label} is not {level}"

    return FEATURE_LABELS.get(feature_name, _prettify(feature_name))


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
        # Guideline-grounded combination flags
        "Liability_NoPolice", "HighValue_Liability", "Sport_YoungDriver",
        "VeryEarlyClaimFlag", "EarlyClaimFlag", "HighSupplements",
        "ExternalAgent_NoPolice",
        # Composite
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
