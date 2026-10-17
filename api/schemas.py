"""
Pydantic Data Schemas for the EKG Reading Assistant API
Defines request and response payloads for clinical analysis, interval extraction, and classification.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class ECGAnalysisRequest(BaseModel):
    """Input payload for 12-lead ECG analysis."""
    ecg_id: Optional[int] = Field(None, description="Optional PTB-XL record ID for benchmark lookup")
    signal: Optional[List[List[float]]] = Field(
        None,
        description="Raw 12-lead signal as (samples, 12) or (12, samples) array in mV"
    )
    sampling_rate: int = Field(100, description="Signal sampling frequency in Hz (default: 100)")
    patient_age: Optional[float] = Field(None, description="Patient age in years")
    patient_sex: Optional[str] = Field("M", description="Patient sex ('M' or 'F')")
    symptoms: Optional[str] = Field("일상 진료 심전도 검사", description="Chief complaint or symptoms")
    output_format: str = Field("narrative", description="Output report format: 'narrative', 'json', 'both', or 'none'")


class HealthResponse(BaseModel):
    """Health status of the API, ONNX model, and Ollama service."""
    status: str
    onnx_model_loaded: bool
    ollama_connected: bool
    ptbxl_dataset_available: bool
    version: str = "1.0.0"


class FeatureExtractionResponse(BaseModel):
    """Quantitative physiological features and intervals."""
    heart_rate_bpm: float
    rhythm_regular: bool
    pr_interval_ms: float
    qrs_duration_ms: float
    qt_interval_ms: float
    qtc_interval_ms: float
    heart_axis_deg: float
    heart_axis_category: str
    sokolow_lyon_mv: float
    lvh_voltage_criteria: bool
    st_deviations_mv: Dict[str, float]
    st_elevation_leads: List[str]
    st_depression_leads: List[str]
    stemi_suspected: bool
    wide_qrs: bool
    prolonged_pr: bool
    prolonged_qtc: bool


class ClassificationResponse(BaseModel):
    """1D-ResNet multi-label diagnostic probabilities and urgency."""
    probabilities: Dict[str, float]
    predictions: Dict[str, bool]
    positive_findings: List[str]
    primary_findings: List[str] = Field(default_factory=list, description="Primary clinical pathologies")
    reference_findings: List[str] = Field(default_factory=list, description="Incidental reference findings (PAC, LVH)")
    top_findings: List[Tuple[str, float]]
    clinical_urgency: str


class ECGAnalysisResponse(BaseModel):
    """Comprehensive end-to-end response for EMR integration."""
    status: str
    patient_info: Dict[str, Any]
    quantitative_features: FeatureExtractionResponse
    deep_learning_classification: ClassificationResponse
    ollama_narrative_report: Optional[str] = None
    emr_structured_json: Optional[Dict[str, Any]] = None
    latency_ms: Dict[str, float]
    disclaimer: str
