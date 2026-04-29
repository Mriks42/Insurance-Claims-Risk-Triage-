"""
Data Validation Module
======================
Pandera schema checks for fraud_oracle.csv.
Runs before any preprocessing to catch bad data early.

Usage:
    from data_validation import validate_raw_data
    validate_raw_data(df)          # raises SchemaError if data is invalid
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# Schema definition
# ============================================================
RAW_DATA_SCHEMA = DataFrameSchema(
    columns={
        # ── Identifiers ──────────────────────────────────────
        "PolicyNumber": Column(
            int,
            checks=Check.greater_than(0),
            nullable=False,
            description="Unique policy identifier",
        ),

        # ── Target ───────────────────────────────────────────
        "FraudFound_P": Column(
            int,
            checks=Check.isin([0, 1]),
            nullable=False,
            description="Binary fraud label",
        ),

        # ── Numeric features ─────────────────────────────────
        "Age": Column(
            int,
            checks=[Check.greater_than_or_equal_to(16),
                    Check.less_than_or_equal_to(100)],
            nullable=False,
        ),
        "Deductible": Column(
            int,
            checks=Check.isin([300, 400, 500, 700]),
            nullable=False,
        ),
        "WeekOfMonth": Column(
            int,
            checks=[Check.greater_than_or_equal_to(1),
                    Check.less_than_or_equal_to(5)],
            nullable=False,
        ),
        "WeekOfMonthClaimed": Column(
            int,
            checks=[Check.greater_than_or_equal_to(1),
                    Check.less_than_or_equal_to(5)],
            nullable=False,
        ),
        "RepNumber": Column(int, nullable=False),
        "DriverRating": Column(
            int,
            checks=[Check.greater_than_or_equal_to(1),
                    Check.less_than_or_equal_to(4)],
            nullable=False,
        ),
        "Year": Column(int, nullable=False),

        # ── Categorical features ──────────────────────────────
        "Month": Column(
            str,
            checks=Check.isin([
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]),
            nullable=False,
        ),
        "DayOfWeek": Column(
            str,
            checks=Check.isin([
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]),
            nullable=False,
        ),
        "DayOfWeekClaimed": Column(
            str,
            checks=Check.isin([
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday",
            ]),
            nullable=False,
        ),
        "MonthClaimed": Column(
            str,
            checks=Check.isin([
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ]),
            nullable=False,
        ),
        "Sex": Column(
            str,
            checks=Check.isin(["Male", "Female"]),
            nullable=False,
        ),
        "MaritalStatus": Column(
            str,
            checks=Check.isin(["Single", "Married", "Widow", "Divorced"]),
            nullable=False,
        ),
        "Fault": Column(
            str,
            checks=Check.isin(["Policy Holder", "Third Party"]),
            nullable=False,
        ),
        "VehicleCategory": Column(
            str,
            checks=Check.isin(["Sedan", "Sport", "Utility"]),
            nullable=False,
        ),
        "BasePolicy": Column(
            str,
            checks=Check.isin(["Liability", "Collision", "All Perils"]),
            nullable=False,
        ),
        "PoliceReportFiled": Column(
            str,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "WitnessPresent": Column(
            str,
            checks=Check.isin(["Yes", "No"]),
            nullable=False,
        ),
        "AgentType": Column(
            str,
            checks=Check.isin(["External", "Internal"]),
            nullable=False,
        ),
        "AccidentArea": Column(
            str,
            checks=Check.isin(["Urban", "Rural"]),
            nullable=False,
        ),
        "PastNumberOfClaims": Column(
            str,
            checks=Check.isin(["none", "1", "2 to 4", "more than 4"]),
            nullable=False,
        ),
        "AgeOfVehicle": Column(
            str,
            checks=Check.isin([
                "new", "2 years", "3 years", "4 years",
                "5 years", "6 years", "7 years", "more than 7",
            ]),
            nullable=False,
        ),
        "AgeOfPolicyHolder": Column(
            str,
            checks=Check.isin([
                "16 to 17", "18 to 20", "21 to 25", "26 to 30",
                "31 to 35", "36 to 40", "41 to 50", "51 to 65", "over 65",
            ]),
            nullable=False,
        ),
        "NumberOfSuppliments": Column(
            str,
            checks=Check.isin(["none", "1 to 2", "3 to 5", "more than 5"]),
            nullable=False,
        ),
        "AddressChange_Claim": Column(
            str,
            checks=Check.isin([
                "no change", "under 6 months", "1 year",
                "2 to 3 years", "4 to 8 years",
            ]),
            nullable=False,
        ),
        "NumberOfCars": Column(
            str,
            checks=Check.isin([
                "1 vehicle", "2 vehicles", "3 to 4",
                "5 to 8", "more than 8",
            ]),
            nullable=False,
        ),
        "Days_Policy_Accident": Column(
            str,
            checks=Check.isin([
                "none", "1 to 7", "8 to 15", "15 to 30", "more than 30",
            ]),
            nullable=False,
        ),
        "Days_Policy_Claim": Column(
            str,
            checks=Check.isin([
                "none", "1 to 7", "8 to 15", "15 to 30", "more than 30",
            ]),
            nullable=False,
        ),
        "VehiclePrice": Column(
            str,
            checks=Check.isin([
                "less than 20000", "20000 to 29000", "30000 to 39000",
                "40000 to 59000", "60000 to 69000", "more than 69000",
            ]),
            nullable=False,
        ),
    },
    checks=[
        # Dataset-level checks
        Check(lambda df: len(df) > 100,
              error="Dataset has fewer than 100 rows — likely wrong file"),
        Check(lambda df: df["FraudFound_P"].mean() < 0.5,
              error="Fraud rate above 50% — check target column"),
    ],
    coerce=False,
    strict=False,   # allow extra columns (e.g. Make, PolicyType)
)


# ============================================================
# Validation function
# ============================================================
def validate_raw_data(df: pd.DataFrame, verbose: bool = True) -> bool:
    """
    Validate the raw fraud_oracle dataframe against the schema.

    Args:
        df: Raw dataframe loaded from fraud_oracle.csv
        verbose: Print validation summary

    Returns:
        True if valid

    Raises:
        pandera.errors.SchemaError if validation fails
    """
    if verbose:
        print("Running data validation...")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Fraud rate: {df['FraudFound_P'].mean():.4f}")

    try:
        RAW_DATA_SCHEMA.validate(df, lazy=True)
        if verbose:
            print("  [OK] All validation checks passed.")
        return True

    except pa.errors.SchemaErrors as e:
        print("\n[VALIDATION FAILED] Data quality issues found:")
        print(e.failure_cases.to_string(index=False))
        raise


def validate_and_report(df: pd.DataFrame) -> dict:
    """
    Run validation and return a summary report dict (for dashboard display).
    Does not raise — returns pass/fail per check.
    """
    report = {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "fraud_rate": float(df["FraudFound_P"].mean()),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "checks": [],
    }

    checks = [
        ("Row count > 100",        len(df) > 100),
        ("No duplicate rows",      df.duplicated().sum() == 0),
        ("No missing values",      df.isna().sum().sum() == 0),
        ("Fraud rate < 50%",       df["FraudFound_P"].mean() < 0.5),
        ("Age range 16-100",       df["Age"].between(16, 100).all()),
        ("Deductible valid",       df["Deductible"].isin([300, 400, 500, 700]).all()),
        ("Binary target",          df["FraudFound_P"].isin([0, 1]).all()),
        ("Fault values valid",     df["Fault"].isin(["Policy Holder", "Third Party"]).all()),
        ("BasePolicy valid",       df["BasePolicy"].isin(["Liability", "Collision", "All Perils"]).all()),
        ("PoliceReport valid",     df["PoliceReportFiled"].isin(["Yes", "No"]).all()),
    ]

    for name, result in checks:
        report["checks"].append({
            "check": name,
            "passed": bool(result),
            "status": "✅ Pass" if result else "❌ Fail",
        })

    report["all_passed"] = all(c["passed"] for c in report["checks"])
    return report


# ============================================================
# Standalone run
# ============================================================
if __name__ == "__main__":
    import pandas as pd
    from config import DATA_PATH

    df = pd.read_csv(DATA_PATH)
    report = validate_and_report(df)

    print("\n" + "=" * 50)
    print("DATA VALIDATION REPORT")
    print("=" * 50)
    print(f"Rows:          {report['total_rows']:,}")
    print(f"Columns:       {report['total_cols']}")
    print(f"Fraud rate:    {report['fraud_rate']:.4f}")
    print(f"Missing vals:  {report['missing_values']}")
    print(f"Duplicates:    {report['duplicate_rows']}")
    print("\nChecks:")
    for c in report["checks"]:
        print(f"  {c['status']}  {c['check']}")
    print(f"\nOverall: {'ALL PASSED' if report['all_passed'] else 'SOME FAILED'}")
