"""
Discharge Planning Agent - FastAPI Application

This module provides the REST API for the discharge planning service.
It exposes endpoints for analyzing patient discharge readiness with
full explainability.

================================================================================
API DESIGN FOR CLINICAL DECISION SUPPORT
================================================================================

This API is designed for integration with:
1. Hospital EMR systems (Epic, Cerner) via HL7/FHIR interfaces
2. Case management dashboards for discharge coordinators
3. Charge nurse workflows for bed management
4. The central orchestrator for multi-agent coordination

Key Design Principles:
─────────────────────
1. EXPLAINABILITY: Every recommendation includes detailed reasoning
2. AUDIT TRAIL: All assessments are logged with timestamps
3. NO AUTONOMOUS ACTION: API provides recommendations, not orders
4. BATCH SUPPORT: Analyze all inpatients efficiently for rounds

================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from config import settings
from model import (
    DischargeAnalyzer,
    DischargeAssessment,
    DischargeRecommendation,
    RuleVeto,
    VetoSeverity,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class VitalSigns(BaseModel):
    """Vital signs data for a patient."""
    
    temperature_celsius: Optional[float] = Field(
        default=None,
        ge=34.0,
        le=43.0,
        description="Body temperature in Celsius"
    )
    heart_rate: Optional[int] = Field(
        default=None,
        ge=20,
        le=250,
        description="Heart rate in beats per minute"
    )
    systolic_bp: Optional[int] = Field(
        default=None,
        ge=50,
        le=300,
        description="Systolic blood pressure in mmHg"
    )
    diastolic_bp: Optional[int] = Field(
        default=None,
        ge=20,
        le=200,
        description="Diastolic blood pressure in mmHg"
    )
    spo2: Optional[float] = Field(
        default=None,
        ge=50.0,
        le=100.0,
        description="Oxygen saturation percentage"
    )
    respiratory_rate: Optional[int] = Field(
        default=None,
        ge=5,
        le=60,
        description="Respiratory rate in breaths per minute"
    )


class LabValues(BaseModel):
    """Laboratory values for a patient."""
    
    wbc_count: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="White blood cell count (x10^9/L)"
    )
    hemoglobin: Optional[float] = Field(
        default=None,
        ge=2.0,
        le=25.0,
        description="Hemoglobin level (g/dL)"
    )
    creatinine: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Creatinine level (mg/dL)"
    )
    potassium: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=10.0,
        description="Potassium level (mEq/L)"
    )
    sodium: Optional[float] = Field(
        default=None,
        ge=100.0,
        le=180.0,
        description="Sodium level (mEq/L)"
    )
    glucose: Optional[float] = Field(
        default=None,
        ge=10.0,
        le=1000.0,
        description="Blood glucose (mg/dL)"
    )


class ClinicalStatus(BaseModel):
    """Clinical status indicators for a patient."""
    
    pending_labs: bool = Field(
        default=False,
        description="Whether patient has pending lab results"
    )
    pending_imaging: bool = Field(
        default=False,
        description="Whether patient has pending imaging studies"
    )
    fever_last_24h: bool = Field(
        default=False,
        description="Whether patient had fever in last 24 hours"
    )
    on_iv_medications: bool = Field(
        default=False,
        description="Whether patient is receiving IV medications"
    )
    has_foley_catheter: bool = Field(
        default=False,
        description="Whether patient has indwelling urinary catheter"
    )
    has_wound_vac: bool = Field(
        default=False,
        description="Whether patient has wound VAC in place"
    )
    needs_home_oxygen: bool = Field(
        default=False,
        description="Whether patient will need home oxygen"
    )


class FunctionalStatus(BaseModel):
    """Functional status indicators for a patient."""
    
    mobility_score: int = Field(
        default=5,
        ge=0,
        le=10,
        description="Physical therapy mobility score (0-10)"
    )
    pain_score: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Current pain level (0-10)"
    )
    fall_risk_score: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Fall risk assessment score (0-5)"
    )
    ambulatory_status: bool = Field(
        default=True,
        description="Whether patient can ambulate independently"
    )
    tolerating_oral_intake: bool = Field(
        default=True,
        description="Whether patient is tolerating oral diet"
    )


class DischargeRequirements(BaseModel):
    """Discharge preparation requirements status."""
    
    pharmacy_reconciliation_complete: bool = Field(
        default=False,
        description="Whether pharmacy medication reconciliation is complete"
    )
    social_work_cleared: bool = Field(
        default=True,
        description="Whether social work has cleared for discharge"
    )
    needs_social_work: bool = Field(
        default=False,
        description="Whether patient requires social work evaluation"
    )
    has_caregiver_at_home: bool = Field(
        default=True,
        description="Whether patient has caregiver support at home"
    )


class PatientAnalysisRequest(BaseModel):
    """
    Request schema for single patient discharge analysis.
    
    This comprehensive schema captures all clinical data needed for
    discharge readiness assessment.
    """
    
    patient_id: str = Field(
        ...,
        description="Unique patient identifier"
    )
    admission_datetime: datetime = Field(
        ...,
        description="Date and time of hospital admission"
    )
    admission_type: str = Field(
        default="medical",
        description="Type of admission (emergency, surgical, medical, observation)"
    )
    primary_diagnosis: Optional[str] = Field(
        default=None,
        description="Primary diagnosis (ICD-10 code or description)"
    )
    age: int = Field(
        default=50,
        ge=0,
        le=130,
        description="Patient age in years"
    )
    
    # Nested clinical data
    vital_signs: VitalSigns = Field(
        default_factory=VitalSigns,
        description="Current vital signs"
    )
    lab_values: LabValues = Field(
        default_factory=LabValues,
        description="Most recent laboratory values"
    )
    clinical_status: ClinicalStatus = Field(
        default_factory=ClinicalStatus,
        description="Clinical status indicators"
    )
    functional_status: FunctionalStatus = Field(
        default_factory=FunctionalStatus,
        description="Functional status indicators"
    )
    discharge_requirements: DischargeRequirements = Field(
        default_factory=DischargeRequirements,
        description="Discharge preparation status"
    )
    
    # Comorbidity and history
    charlson_comorbidity_index: int = Field(
        default=2,
        ge=0,
        le=37,
        description="Charlson Comorbidity Index score"
    )
    num_active_diagnoses: int = Field(
        default=3,
        ge=0,
        le=50,
        description="Number of active diagnoses"
    )
    num_procedures_48h: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Number of procedures in last 48 hours"
    )
    prior_readmissions_30d: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of readmissions in prior 30 days"
    )
    
    # Computed field: vital stability
    vital_stability_score: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Stability of vitals over last 24h (0-100)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "P12345",
                "admission_datetime": "2024-01-10T14:30:00Z",
                "admission_type": "medical",
                "primary_diagnosis": "J18.9 - Pneumonia",
                "age": 67,
                "vital_signs": {
                    "temperature_celsius": 37.2,
                    "heart_rate": 78,
                    "systolic_bp": 128,
                    "diastolic_bp": 76,
                    "spo2": 96.0,
                    "respiratory_rate": 16
                },
                "lab_values": {
                    "wbc_count": 8.5,
                    "hemoglobin": 12.1,
                    "creatinine": 0.9,
                    "potassium": 4.2,
                    "sodium": 140,
                    "glucose": 105
                },
                "clinical_status": {
                    "pending_labs": False,
                    "pending_imaging": False,
                    "fever_last_24h": False,
                    "on_iv_medications": False
                },
                "functional_status": {
                    "mobility_score": 8,
                    "pain_score": 2,
                    "ambulatory_status": True,
                    "tolerating_oral_intake": True
                },
                "discharge_requirements": {
                    "pharmacy_reconciliation_complete": True,
                    "social_work_cleared": True
                }
            }
        }
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert nested structure to flat dictionary for model input."""
        flat = {
            "patient_id": self.patient_id,
            "admission_datetime": self.admission_datetime,
            "admission_type": self.admission_type,
            "primary_diagnosis": self.primary_diagnosis,
            "age": self.age,
            "charlson_comorbidity_index": self.charlson_comorbidity_index,
            "num_active_diagnoses": self.num_active_diagnoses,
            "num_procedures_48h": self.num_procedures_48h,
            "prior_readmissions_30d": self.prior_readmissions_30d,
            "vital_stability_score": self.vital_stability_score,
        }
        
        # Flatten vital signs
        if self.vital_signs:
            for key, value in self.vital_signs.model_dump().items():
                flat[key] = value
        
        # Flatten lab values
        if self.lab_values:
            for key, value in self.lab_values.model_dump().items():
                flat[key] = value
        
        # Flatten clinical status
        if self.clinical_status:
            for key, value in self.clinical_status.model_dump().items():
                flat[key] = value
        
        # Flatten functional status
        if self.functional_status:
            for key, value in self.functional_status.model_dump().items():
                flat[key] = value
        
        # Flatten discharge requirements
        if self.discharge_requirements:
            for key, value in self.discharge_requirements.model_dump().items():
                flat[key] = value
        
        # Calculate LOS in hours
        flat["los_hours"] = (
            datetime.utcnow() - self.admission_datetime
        ).total_seconds() / 3600
        
        return flat


class BatchAnalysisRequest(BaseModel):
    """Request schema for batch patient analysis."""
    
    patients: List[PatientAnalysisRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of patients to analyze"
    )
    include_all_details: bool = Field(
        default=True,
        description="Whether to include full analysis details"
    )


class RuleVetoResponse(BaseModel):
    """Response schema for a rule veto."""
    
    rule_id: str
    rule_name: str
    reason: str
    severity: str
    actual_value: Optional[Any] = None
    threshold_value: Optional[Any] = None
    recommendation: Optional[str] = None


class ContributingFactorResponse(BaseModel):
    """Response schema for a contributing factor."""
    
    factor_name: str
    contribution: float
    description: str
    raw_value: Optional[Any] = None


class PatientAnalysisResponse(BaseModel):
    """
    Response schema for discharge analysis.
    
    Provides comprehensive explainability for clinical decision support.
    """
    
    # Identifiers
    request_id: str
    patient_id: str
    assessment_timestamp: datetime
    
    # Final recommendation
    recommendation: str = Field(
        description="LIKELY_READY, NEEDS_REVIEW, NOT_READY, or VETOED"
    )
    recommendation_explanation: str = Field(
        description="Human-readable explanation of the recommendation"
    )
    
    # ML model output
    ml_readiness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="ML model readiness score (0.0-1.0)"
    )
    ml_score_interpretation: str
    
    # Rule engine output
    rule_vetoes: List[RuleVetoResponse]
    rules_passed: int
    rules_failed: int
    
    # Explainability
    contributing_factors: List[ContributingFactorResponse]
    
    # Context
    length_of_stay_hours: Optional[float] = None
    admission_type: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    
    # Metadata
    model_version: str
    rule_engine_version: str


class BatchAnalysisResponse(BaseModel):
    """Response schema for batch analysis."""
    
    request_id: str
    analyzed_at: datetime
    total_patients: int
    
    # Summary statistics
    summary: Dict[str, int] = Field(
        description="Count of patients in each recommendation category"
    )
    
    # Patient results
    patients: List[PatientAnalysisResponse]
    
    # Sorted lists for workflow
    discharge_candidates: List[str] = Field(
        description="Patient IDs that are likely ready for discharge"
    )
    needs_review: List[str] = Field(
        description="Patient IDs that need case manager review"
    )


class HealthResponse(BaseModel):
    """Response schema for health checks."""
    
    status: str
    service: str
    version: str
    timestamp: datetime
    checks: Dict[str, Dict[str, Any]]


# =============================================================================
# APPLICATION STATE
# =============================================================================

class AppState:
    """
    Application state management.
    
    Holds the discharge analyzer and tracks analysis metrics.
    """
    
    def __init__(self):
        self.analyzer: Optional[DischargeAnalyzer] = None
        self.model_loaded_at: Optional[datetime] = None
        self.analyses_performed: int = 0
        self._lock = asyncio.Lock()
    
    async def get_analyzer(self) -> DischargeAnalyzer:
        """Get or initialize the discharge analyzer."""
        async with self._lock:
            if self.analyzer is None:
                self.analyzer = DischargeAnalyzer()
                self.model_loaded_at = datetime.utcnow()
                logger.info("DischargeAnalyzer initialized")
            return self.analyzer


# Global application state
app_state = AppState()


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    
    # Initialize analyzer at startup
    try:
        await app_state.get_analyzer()
    except Exception as e:
        logger.warning(f"Could not initialize analyzer at startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down discharge planning agent")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Discharge Planning Agent",
    description="""
    Hospital discharge readiness assessment service using hybrid ML + Clinical Rules.
    
    ## Overview
    This agent evaluates inpatients to identify those who are clinically safe for 
    discharge, optimizing bed turnover while maintaining patient safety.
    
    ## Key Features
    - **Hybrid Architecture**: Combines XGBoost ML model with explicit clinical rules
    - **Full Explainability**: Every recommendation includes detailed reasoning
    - **Safety Guardrails**: Critical clinical rules can VETO high ML scores
    - **Batch Support**: Analyze all inpatients efficiently for discharge rounds
    
    ## API Endpoints
    - `POST /analyze`: Batch analysis of all provided inpatients
    - `POST /analyze-single`: Detailed breakdown for one patient
    - `GET /health`: Service health check
    
    ## Clinical Integration
    This service is designed for clinical decision SUPPORT, not autonomous action.
    Final discharge decisions must be made by the attending physician.
    
    ## Why Hybrid Approach?
    Pure ML ("Black Box") models are insufficient for discharge decisions because:
    1. **Legal Liability**: Courts require explainable decision trails
    2. **Regulatory Compliance**: CMS/Joint Commission require documented criteria
    3. **Rare Events**: ML may miss rare but critical contraindications
    4. **Clinical Trust**: Providers need to understand recommendations
    """,
    version=settings.service_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_recommendation_explanation(assessment: DischargeAssessment) -> str:
    """Generate a detailed explanation of the recommendation."""
    rec = assessment.recommendation
    
    if rec == DischargeRecommendation.LIKELY_READY:
        return (
            f"Patient appears ready for discharge based on ML readiness score "
            f"({assessment.ml_readiness_score:.0%}) and all clinical safety rules passed. "
            f"Recommend physician review for final discharge decision."
        )
    
    elif rec == DischargeRecommendation.NEEDS_REVIEW:
        issues = []
        if assessment.rule_vetoes:
            issues.append(f"{len(assessment.rule_vetoes)} clinical rule warning(s)")
        if assessment.ml_readiness_score < settings.readiness_score_high_threshold:
            issues.append(f"moderate ML readiness score ({assessment.ml_readiness_score:.0%})")
        
        return (
            f"Patient needs case manager review before discharge consideration. "
            f"Concerns: {', '.join(issues)}. Review specific issues below."
        )
    
    elif rec == DischargeRecommendation.VETOED:
        critical_vetoes = [v for v in assessment.rule_vetoes if v.severity == VetoSeverity.CRITICAL]
        veto_names = [v.rule_name for v in critical_vetoes[:3]]
        
        return (
            f"Discharge BLOCKED by {len(critical_vetoes)} critical clinical rule(s): "
            f"{', '.join(veto_names)}. Even though ML score was {assessment.ml_readiness_score:.0%}, "
            f"patient safety rules require these issues to be resolved first."
        )
    
    else:  # NOT_READY
        return (
            f"Patient not ready for discharge. ML readiness score "
            f"({assessment.ml_readiness_score:.0%}) indicates continued inpatient care needed. "
            f"Focus on acute medical management."
        )


def _build_patient_response(
    assessment: DischargeAssessment,
    request_id: str,
) -> PatientAnalysisResponse:
    """Build API response from internal assessment."""
    return PatientAnalysisResponse(
        request_id=request_id,
        patient_id=assessment.patient_id,
        assessment_timestamp=assessment.assessment_timestamp,
        recommendation=assessment.recommendation.value,
        recommendation_explanation=_generate_recommendation_explanation(assessment),
        ml_readiness_score=round(assessment.ml_readiness_score, 3),
        ml_score_interpretation=assessment.ml_score_interpretation,
        rule_vetoes=[
            RuleVetoResponse(
                rule_id=v.rule_id,
                rule_name=v.rule_name,
                reason=v.reason,
                severity=v.severity.value,
                actual_value=v.actual_value,
                threshold_value=v.threshold_value,
                recommendation=v.recommendation,
            )
            for v in assessment.rule_vetoes
        ],
        rules_passed=assessment.rules_passed,
        rules_failed=assessment.rules_failed,
        contributing_factors=[
            ContributingFactorResponse(
                factor_name=f.factor_name,
                contribution=round(f.contribution, 3),
                description=f.description,
                raw_value=f.raw_value,
            )
            for f in assessment.contributing_factors
        ],
        length_of_stay_hours=round(assessment.length_of_stay_hours, 1) if assessment.length_of_stay_hours else None,
        admission_type=assessment.admission_type,
        primary_diagnosis=assessment.primary_diagnosis,
        model_version=assessment.model_version,
        rule_engine_version=assessment.rule_engine_version,
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for Kubernetes probes.
    
    Checks service status, model availability, and rule engine status.
    """
    checks = {}
    overall_status = "healthy"
    
    # Check analyzer availability
    analyzer_check = {"status": "ok"}
    try:
        analyzer = await app_state.get_analyzer()
        analyzer_check["loaded_at"] = app_state.model_loaded_at.isoformat() if app_state.model_loaded_at else None
        analyzer_check["analyses_performed"] = app_state.analyses_performed
        analyzer_check["ml_model_fitted"] = analyzer.ml_model.is_fitted
        analyzer_check["rule_engine_version"] = analyzer.rule_engine.VERSION
    except Exception as e:
        analyzer_check["status"] = "error"
        analyzer_check["message"] = str(e)
        overall_status = "unhealthy"
    checks["analyzer"] = analyzer_check
    
    return HealthResponse(
        status=overall_status,
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.utcnow(),
        checks=checks,
    )


@app.post(
    "/analyze",
    response_model=BatchAnalysisResponse,
    tags=["Analysis"],
    summary="Batch analyze inpatients for discharge readiness"
)
async def analyze_batch(request: BatchAnalysisRequest) -> BatchAnalysisResponse:
    """
    Analyze multiple patients for discharge readiness.
    
    This endpoint is designed for:
    - Morning discharge rounds (analyze all unit patients)
    - Bed management dashboards (identify turnover opportunities)
    - Case manager workflows (prioritize reviews)
    
    **Returns:**
    - Summary statistics of readiness categories
    - Detailed analysis for each patient
    - Sorted lists of discharge candidates and review-needed patients
    
    **Performance:**
    - Typical analysis time: ~100ms per patient
    - Recommended batch size: 50-100 patients
    - Maximum batch size: 500 patients
    """
    request_id = str(uuid.uuid4())
    logger.info(
        f"Batch analysis request: {request_id}",
        extra={"n_patients": len(request.patients)}
    )
    
    analyzer = await app_state.get_analyzer()
    
    # Convert requests to flat dictionaries
    patients_data = [p.to_flat_dict() for p in request.patients]
    
    # Run batch analysis
    assessments = analyzer.analyze_batch(patients_data)
    
    # Update analytics
    app_state.analyses_performed += len(assessments)
    
    # Build responses
    patient_responses = [
        _build_patient_response(assessment, request_id)
        for assessment in assessments
    ]
    
    # Calculate summary
    summary = {
        "LIKELY_READY": 0,
        "NEEDS_REVIEW": 0,
        "NOT_READY": 0,
        "VETOED": 0,
    }
    discharge_candidates = []
    needs_review = []
    
    for assessment, response in zip(assessments, patient_responses):
        summary[assessment.recommendation.value] += 1
        
        if assessment.recommendation == DischargeRecommendation.LIKELY_READY:
            discharge_candidates.append(assessment.patient_id)
        elif assessment.recommendation == DischargeRecommendation.NEEDS_REVIEW:
            needs_review.append(assessment.patient_id)
    
    logger.info(
        f"Batch analysis complete: {request_id}",
        extra={
            "total": len(assessments),
            "ready": summary["LIKELY_READY"],
            "review": summary["NEEDS_REVIEW"],
            "not_ready": summary["NOT_READY"],
            "vetoed": summary["VETOED"],
        }
    )
    
    return BatchAnalysisResponse(
        request_id=request_id,
        analyzed_at=datetime.utcnow(),
        total_patients=len(assessments),
        summary=summary,
        patients=patient_responses,
        discharge_candidates=discharge_candidates,
        needs_review=needs_review,
    )


@app.post(
    "/analyze-single",
    response_model=PatientAnalysisResponse,
    tags=["Analysis"],
    summary="Analyze single patient with detailed breakdown"
)
async def analyze_single(request: PatientAnalysisRequest) -> PatientAnalysisResponse:
    """
    Analyze a single patient for discharge readiness with detailed breakdown.
    
    This endpoint provides comprehensive analysis including:
    - ML readiness score with contributing factors
    - All clinical rule evaluations with explanations
    - Specific recommendations for any rule violations
    
    **Use Cases:**
    - Detailed review of specific patient
    - Understanding why a patient was flagged
    - Documentation for discharge planning discussions
    
    **Example Response:**
    ```json
    {
        "recommendation": "VETOED",
        "recommendation_explanation": "Discharge BLOCKED by 1 critical rule: FEVER_CHECK...",
        "ml_readiness_score": 0.82,
        "rule_vetoes": [
            {
                "rule_name": "FEVER_CHECK",
                "reason": "Temperature 38.7°C exceeds safe threshold (38.0°C)",
                "severity": "CRITICAL",
                "recommendation": "Investigate source of fever before discharge..."
            }
        ],
        "contributing_factors": [
            {
                "factor_name": "mobility_score",
                "contribution": 0.15,
                "description": "Good mobility (8/10) increases readiness"
            }
        ]
    }
    ```
    """
    request_id = str(uuid.uuid4())
    
    logger.info(
        f"Single patient analysis: {request_id}",
        extra={"patient_id": request.patient_id}
    )
    
    analyzer = await app_state.get_analyzer()
    
    # Convert to flat dictionary
    patient_data = request.to_flat_dict()
    
    # Run analysis
    assessment = analyzer.analyze_patient(patient_data)
    
    # Update analytics
    app_state.analyses_performed += 1
    
    # Build response
    response = _build_patient_response(assessment, request_id)
    
    logger.info(
        f"Analysis complete for {request.patient_id}: {assessment.recommendation.value}",
        extra={
            "ml_score": assessment.ml_readiness_score,
            "vetoes": len(assessment.rule_vetoes),
        }
    )
    
    return response


@app.get(
    "/rules",
    tags=["Information"],
    summary="List all clinical rules in the rule engine"
)
async def list_rules() -> Dict[str, Any]:
    """
    List all clinical rules implemented in the rule engine.
    
    This endpoint provides documentation of:
    - All rule IDs and names
    - Rule categories (vital signs, labs, safety, operational)
    - Severity levels and thresholds
    
    Useful for clinical governance review and EMR integration.
    """
    return {
        "rule_engine_version": "1.0.0",
        "categories": [
            {
                "name": "Vital Signs Rules",
                "rules": [
                    {"id": "VS001", "name": "FEVER_CHECK", "severity": "CRITICAL"},
                    {"id": "VS002", "name": "RECENT_FEVER", "severity": "WARNING"},
                    {"id": "VS003", "name": "TACHYCARDIA_CHECK", "severity": "WARNING"},
                    {"id": "VS004", "name": "BRADYCARDIA_CHECK", "severity": "WARNING"},
                    {"id": "VS005", "name": "HYPOXIA_CHECK", "severity": "CRITICAL"},
                    {"id": "VS006", "name": "HYPOTENSION_CHECK", "severity": "CRITICAL"},
                    {"id": "VS007", "name": "HYPERTENSIVE_CRISIS_CHECK", "severity": "CRITICAL"},
                ]
            },
            {
                "name": "Laboratory Rules",
                "rules": [
                    {"id": "LAB001", "name": "HIGH_WBC_CHECK", "severity": "WARNING"},
                    {"id": "LAB002", "name": "LOW_WBC_CHECK", "severity": "WARNING"},
                    {"id": "LAB003", "name": "SEVERE_ANEMIA_CHECK", "severity": "CRITICAL"},
                    {"id": "LAB004", "name": "KIDNEY_FUNCTION_CHECK", "severity": "WARNING"},
                    {"id": "LAB005", "name": "HYPOKALEMIA_CHECK", "severity": "CRITICAL"},
                    {"id": "LAB006", "name": "HYPERKALEMIA_CHECK", "severity": "CRITICAL"},
                ]
            },
            {
                "name": "Operational Rules",
                "rules": [
                    {"id": "OP001", "name": "PENDING_LABS_CHECK", "severity": "WARNING"},
                    {"id": "OP002", "name": "PENDING_IMAGING_CHECK", "severity": "WARNING"},
                    {"id": "OP003", "name": "MINIMUM_LOS_CHECK", "severity": "WARNING"},
                ]
            },
            {
                "name": "Safety Rules",
                "rules": [
                    {"id": "SF001", "name": "IV_MEDICATION_CHECK", "severity": "WARNING"},
                    {"id": "SF002", "name": "FOLEY_CATHETER_CHECK", "severity": "INFO"},
                    {"id": "SF003", "name": "HIGH_FALL_RISK_CHECK", "severity": "WARNING"},
                    {"id": "SF004", "name": "PHARMACY_RECONCILIATION_CHECK", "severity": "WARNING"},
                    {"id": "SF005", "name": "SOCIAL_WORK_CLEARANCE_CHECK", "severity": "WARNING"},
                ]
            },
        ],
        "thresholds": {
            "fever_threshold_celsius": settings.fever_threshold_celsius,
            "tachycardia_threshold": settings.tachycardia_threshold,
            "bradycardia_threshold": settings.bradycardia_threshold,
            "hypoxia_threshold": settings.hypoxia_threshold,
            "hypotension_systolic_threshold": settings.hypotension_systolic_threshold,
            "hypertension_systolic_threshold": settings.hypertension_systolic_threshold,
            "wbc_high_threshold": settings.wbc_high_threshold,
            "wbc_low_threshold": settings.wbc_low_threshold,
            "hemoglobin_low_threshold": settings.hemoglobin_low_threshold,
            "creatinine_high_threshold": settings.creatinine_high_threshold,
            "potassium_low_threshold": settings.potassium_low_threshold,
            "potassium_high_threshold": settings.potassium_high_threshold,
        }
    }


@app.get(
    "/model/info",
    tags=["Information"],
    summary="Get ML model information"
)
async def get_model_info() -> Dict[str, Any]:
    """
    Get information about the ML readiness model.
    
    Includes:
    - Model training metadata
    - Feature definitions
    - Hyperparameters
    """
    analyzer = await app_state.get_analyzer()
    
    return {
        "model_type": "XGBoost Classifier",
        "is_fitted": analyzer.ml_model.is_fitted,
        "training_metadata": analyzer.ml_model.model_metadata,
        "hyperparameters": {
            "n_estimators": settings.xgb_n_estimators,
            "max_depth": settings.xgb_max_depth,
            "learning_rate": settings.xgb_learning_rate,
            "min_child_weight": settings.xgb_min_child_weight,
        },
        "feature_definitions": analyzer.ml_model.FEATURE_DEFINITIONS,
        "readiness_thresholds": {
            "high_threshold": settings.readiness_score_high_threshold,
            "medium_threshold": settings.readiness_score_medium_threshold,
        }
    }


# =============================================================================
# EXCEPTION HANDLERS
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.debug else None,
        },
    )


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
