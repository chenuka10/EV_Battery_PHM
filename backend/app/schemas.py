from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    engine: Literal["logistic", "xgboost"] = Field(
        default="logistic",
        description="Inference model tier: 'logistic' for Edge BMS Calibrated Logistic Regression, 'xgboost' for Cloud Fleet Analytics Tuned XGBoost."
    )
    tau: float = Field(
        default=0.193,
        ge=0.01,
        le=0.99,
        description="Operational decision cutoff threshold for Task 2 critical failure early warning."
    )
    inputs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of battery telemetry feature values (supports 12 primary slider features or all 70 raw attributes)."
    )


class PredictResponse(BaseModel):
    rul_cycles: float
    p_failure: float
    flagged: bool
    status: Literal["nominal", "early_warning", "critical"]
    status_label: str
    alert_title: str
    alert_action: str
    alert_note: str
    engineered_features: Dict[str, float]
    imputed_fields: List[str]
    warnings: List[str]
    model_names: Dict[str, str]
    version: str
    timestamp: str


class ThresholdMetrics(BaseModel):
    recall: float
    precision: float
    f1: float
    tp: int
    fp: int
    fn: int
    tn: int


class ThresholdQueryResponse(BaseModel):
    engine: str
    tau: float
    metrics: ThresholdMetrics
    curve: Dict[str, List[float]]
    markers: Dict[str, Any]


class OperatingModesResponse(BaseModel):
    engine: str
    modes: Dict[str, Any]


class SimulationRequest(BaseModel):
    n_vehicles: int = Field(default=5000, ge=1, le=1_000_000, description="Total number of fleet EVs.")
    daily_capacity: int = Field(default=300, ge=1, le=50_000, description="Depot monthly inspection throughput capacity.")
    tau: Optional[float] = Field(default=None, ge=0.01, le=0.99, description="Explicit threshold or None to derive from mode.")
    mode: Optional[str] = Field(default=None, description="Operating mode: 'safety', 'balanced', or 'precision'.")
    engine: Literal["logistic", "xgboost"] = Field(default="logistic", description="Inference engine.")


class SimulationResponse(BaseModel):
    fleet_size: int
    capacity: int
    tau_used: float
    mode_used: str
    expected_failures: float
    expected_normal: float
    expected_flags: float
    expected_intercepted: float
    expected_missed: float
    workload_ratio: float
    capacity_status: str
    scaling_assumption: str


class ExplainRequest(BaseModel):
    engine: Literal["logistic", "xgboost"] = Field(default="logistic")
    inputs: Dict[str, Any] = Field(default_factory=dict)


class AttributionItem(BaseModel):
    feature: str
    weight: float
    direction: str
    description: str


class ExplainResponse(BaseModel):
    engine: str
    method: str
    top_risk_factors: List[AttributionItem]
    top_protective_factors: List[AttributionItem]
    caption: str


class HealthResponse(BaseModel):
    status: str
    artifacts: Dict[str, bool]
    library_versions: Dict[str, str]


class DataHealthResponse(BaseModel):
    total_records: int
    features_count: int
    incomplete_records: int
    incomplete_pct: float
    complete_records: int
    duplicate_records: int
    excluded_task1_records: int
    outliers: List[Dict[str, Any]]


class SelfTestItem(BaseModel):
    scenario: str
    transformed_features: int
    rul_cycles: float
    p_failure: float
    status: str
    passed: bool


class SelfTestResponse(BaseModel):
    all_passed: bool
    results: List[SelfTestItem]


class LeaderboardRow(BaseModel):
    model: str
    paradigm: Optional[str] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[str] = None
    recall: Optional[str] = None
    f1: Optional[str] = None
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    missed: Optional[int] = None
    false_alarms: Optional[int] = None
    status: str
    footnote: Optional[str] = None


class LeaderboardsResponse(BaseModel):
    task1: List[LeaderboardRow]
    task2: List[LeaderboardRow]
    footnotes: List[str]
