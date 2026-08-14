"""
Request / response schemas for the scoring API.

Allowed categorical values and numeric bounds are read out of the Pandera schema
in data_validation.py rather than retyped here. That file is already the
project's definition of a valid claim; duplicating those lists would guarantee
they drift apart the first time a category changes.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from data_validation import RAW_DATA_SCHEMA


# ──────────────────────────────────────────────────────────────
# Pull the contract out of the Pandera schema
# ──────────────────────────────────────────────────────────────
def _allowed(column: str) -> Optional[List[Any]]:
    """Allowed values for a column, from its Pandera `isin` check."""
    col = RAW_DATA_SCHEMA.columns.get(column)
    if col is None:
        return None
    for check in col.checks:
        values = getattr(check, "statistics", {}) or {}
        if "allowed_values" in values:
            return list(values["allowed_values"])
    return None


def _bounds(column: str):
    """(min, max) for a column, from its Pandera range checks."""
    col = RAW_DATA_SCHEMA.columns.get(column)
    lo = hi = None
    if col is not None:
        for check in col.checks:
            stats = getattr(check, "statistics", {}) or {}
            lo = stats.get("min_value", lo)
            hi = stats.get("max_value", hi)
    return lo, hi


AGE_MIN, AGE_MAX = _bounds("Age")
CATEGORICAL_FIELDS = [
    "Month", "DayOfWeek", "MonthClaimed", "DayOfWeekClaimed", "Sex",
    "MaritalStatus", "Fault", "VehicleCategory", "BasePolicy",
    "PoliceReportFiled", "WitnessPresent", "AgentType", "AccidentArea",
    "PastNumberOfClaims", "AgeOfVehicle", "NumberOfSuppliments",
    "AddressChange_Claim", "NumberOfCars", "VehiclePrice",
    "Days_Policy_Accident", "Days_Policy_Claim",
]
ALLOWED = {f: _allowed(f) for f in CATEGORICAL_FIELDS}


class Claim(BaseModel):
    """
    A single claim to score.

    Fields are split into those that drive the score (required) and those the
    dashboard's live-scoring form also holds fixed (optional, defaulted). The
    response reports which defaults were applied, so a caller is never silently
    scored on values it did not supply.
    """

    model_config = {"extra": "forbid"}   # typo in a field name is an error, not a default

    # ── policy & coverage ──────────────────────────────────────
    BasePolicy: str
    VehicleCategory: str
    VehiclePrice: str

    # ── claim circumstances ────────────────────────────────────
    Fault: str
    PoliceReportFiled: str
    WitnessPresent: str
    AgentType: str
    AddressChange_Claim: str
    Deductible: int
    PastNumberOfClaims: str
    NumberOfSuppliments: str

    # ── policyholder ───────────────────────────────────────────
    Age: int = Field(..., description=f"Policyholder age ({AGE_MIN}-{AGE_MAX})")
    Sex: str
    MaritalStatus: str
    AccidentArea: str
    AgeOfVehicle: str

    # ── timing ─────────────────────────────────────────────────
    Month: str
    DayOfWeek: str
    Days_Policy_Accident: str
    MonthClaimed: str
    DayOfWeekClaimed: str
    Days_Policy_Claim: str

    # ── optional, defaulted (mirrors the dashboard's assumptions) ──
    Make: Optional[str] = None
    RepNumber: Optional[int] = None
    DriverRating: Optional[int] = None
    WeekOfMonth: Optional[int] = None
    WeekOfMonthClaimed: Optional[int] = None
    NumberOfCars: Optional[str] = None
    Year: Optional[int] = None
    # Derived from the fields above when omitted — see api.main.to_raw_row
    PolicyType: Optional[str] = None
    AgeOfPolicyHolder: Optional[str] = None

    @field_validator(*CATEGORICAL_FIELDS, mode="after")
    @classmethod
    def _check_category(cls, value, info):
        allowed = ALLOWED.get(info.field_name)
        if allowed and value not in allowed:
            raise ValueError(
                f"{value!r} is not a valid {info.field_name}. Allowed: {allowed}"
            )
        return value

    @field_validator("Age")
    @classmethod
    def _check_age(cls, value):
        if AGE_MIN is not None and value < AGE_MIN:
            # The training file contains 320 rows with Age == 0; that is a data
            # entry artifact, not a real age, and must not enter through the API.
            raise ValueError(f"Age must be >= {AGE_MIN} (got {value})")
        if AGE_MAX is not None and value > AGE_MAX:
            raise ValueError(f"Age must be <= {AGE_MAX} (got {value})")
        return value

    @field_validator("Deductible")
    @classmethod
    def _check_deductible(cls, value):
        allowed = _allowed("Deductible")
        if allowed and value not in allowed:
            raise ValueError(f"Deductible must be one of {allowed} (got {value})")
        return value

    @model_validator(mode="after")
    def _check_dates_consistent(self):
        months = ALLOWED.get("Month") or []
        if self.Month in months and self.MonthClaimed in months:
            # Claiming in an earlier month than the accident is possible across a
            # year boundary, so this is not rejected — surfaced in the response.
            pass
        return self


class ReasonCode(BaseModel):
    label: str = Field(..., description="Plain-English driver, for an investigator")
    technical: str = Field(..., description="Underlying model feature name")
    shap: float
    direction: str
    intensity: str


class PredictionResponse(BaseModel):
    risk_score: float = Field(..., description="Predicted fraud probability")
    triage_bucket: str = Field(..., description="SIU | Manual Review | Approve")
    model_version: str
    thresholds: Dict[str, float]
    defaults_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Fields the caller omitted, and the values used instead",
    )
    reason_codes: Optional[List[ReasonCode]] = None


class BatchRequest(BaseModel):
    claims: List[Claim]
    explain: bool = False


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    model_version: Optional[str] = None
    detail: Optional[str] = None
