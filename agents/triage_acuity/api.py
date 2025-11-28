"""
Triage & Acuity Agent - FastAPI Application

This module provides the REST API for the AI-powered triage service.
It exposes endpoints for triage predictions, clinician overrides, and health checks.

================================================================================
API DESIGN FOR CLINICAL SAFETY
================================================================================

The API is designed with clinical safety as the primary concern:

1. RED FLAG OVERRIDE:
   ───────────────────
   Before ML prediction, symptoms text is scanned for Red Flags.
   If detected, acuity is OVERRIDDEN to 1 or 2 regardless of ML output.
   This ensures critical patients are never under-triaged.

2. CLINICIAN OVERRIDE ENDPOINT:
   ─────────────────────────────
   Allows doctors/nurses to correct AI predictions.
   Override data is stored for retraining (Active Learning).
   This creates a continuous improvement feedback loop.

3. AUDIT LOGGING:
   ────────────────
   All predictions and overrides are logged for:
   - Clinical quality review
   - Medicolegal documentation
   - Model performance monitoring

4. CONFIDENCE SCORES:
   ───────────────────
   Each prediction includes confidence scores.
   Low confidence predictions are flagged for human review.

================================================================================
CONTINUOUS LEARNING THROUGH CLINICIAN OVERRIDES
================================================================================

The /override endpoint is CRUCIAL for the Active Learning loop:

    ┌─────────────────┐
    │ Patient Arrives │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ AI Triage       │───────────────────────┐
    │ Prediction      │                       │
    └────────┬────────┘                       │
             │                                │
             ▼                                ▼
    ┌─────────────────┐             ┌─────────────────┐
    │ Clinician       │             │ Correct?        │
    │ Reviews         │             │                 │
    └────────┬────────┘             └────────┬────────┘
             │                                │
             ▼                                ▼
      ┌──────┴──────┐                   ┌─────┴─────┐
      │             │                   │           │
      ▼             ▼                   ▼           ▼
    ┌─────┐     ┌─────────┐        ┌─────────┐ ┌─────────┐
    │ YES │     │ NO      │        │ Accept  │ │Override │
    │     │     │ Override│        │         │ │ POST    │
    └──┬──┘     └────┬────┘        └─────────┘ └────┬────┘
       │             │                              │
       │             └──────────────────────────────┤
       │                                            │
       ▼                                            ▼
    ┌─────────────────┐                    ┌─────────────────┐
    │ Proceed with    │                    │ Save Override   │
    │ Care            │                    │ for Retraining  │
    └─────────────────┘                    └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │ FL Training     │
                                           │ Uses Override   │
                                           │ as Ground Truth │
                                           └─────────────────┘

WHY OVERRIDES ARE VALUABLE:
───────────────────────────
1. Ground Truth: Clinician judgment is the gold standard
2. Edge Cases: Captures unusual presentations ML may miss
3. Local Patterns: Hospital-specific patient populations
4. Continuous Improvement: Model learns from corrections
5. Trust Building: Clinicians see their feedback matters

================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from config import settings, AcuityLevel
from model import TriageClassifier
from text_parser import (
    RedFlagParser,
    ParseResult,
    VitalSigns,
    get_parser,
    parse_symptoms,
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

class VitalsInput(BaseModel):
    """Vital signs input for triage."""
    
    heart_rate: Optional[float] = Field(
        default=None,
        ge=20,
        le=250,
        description="Heart rate in beats per minute",
    )
    systolic_bp: Optional[float] = Field(
        default=None,
        ge=40,
        le=300,
        description="Systolic blood pressure in mmHg",
    )
    diastolic_bp: Optional[float] = Field(
        default=None,
        ge=20,
        le=200,
        description="Diastolic blood pressure in mmHg",
    )
    respiratory_rate: Optional[float] = Field(
        default=None,
        ge=4,
        le=60,
        description="Respiratory rate in breaths per minute",
    )
    oxygen_saturation: Optional[float] = Field(
        default=None,
        ge=50,
        le=100,
        description="Oxygen saturation (SpO2) percentage",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=30,
        le=45,
        description="Body temperature in Celsius",
    )
    gcs: Optional[int] = Field(
        default=None,
        ge=3,
        le=15,
        description="Glasgow Coma Scale score (3-15)",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "heart_rate": 88,
                "systolic_bp": 142,
                "diastolic_bp": 88,
                "respiratory_rate": 18,
                "oxygen_saturation": 97,
                "temperature": 37.2,
                "gcs": 15,
            }
        }


class PatientInfo(BaseModel):
    """Patient demographic information."""
    
    age: Optional[int] = Field(
        default=None,
        ge=0,
        le=120,
        description="Patient age in years",
    )
    gender: Optional[str] = Field(
        default=None,
        description="Patient gender (male/female/other)",
    )
    arrival_mode: Optional[str] = Field(
        default="walk_in",
        description="How patient arrived (walk_in, ambulance, helicopter)",
    )


class TriageRequest(BaseModel):
    """Request schema for triage predictions."""
    
    symptoms: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Free-text description of patient symptoms/chief complaint",
    )
    vitals: Optional[VitalsInput] = Field(
        default=None,
        description="Patient vital signs",
    )
    patient: Optional[PatientInfo] = Field(
        default=None,
        description="Patient demographic information",
    )
    pain_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=10,
        description="Self-reported pain score (0-10)",
    )
    chief_complaint_category: Optional[str] = Field(
        default=None,
        description="Categorized chief complaint",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": "Patient complains of severe chest pain radiating to left arm, started 30 minutes ago. Also reports shortness of breath and nausea.",
                "vitals": {
                    "heart_rate": 102,
                    "systolic_bp": 158,
                    "diastolic_bp": 94,
                    "respiratory_rate": 22,
                    "oxygen_saturation": 94,
                    "temperature": 37.0,
                    "gcs": 15,
                },
                "patient": {
                    "age": 58,
                    "gender": "male",
                    "arrival_mode": "ambulance",
                },
                "pain_score": 8,
                "chief_complaint_category": "chest_pain",
            }
        }


class RedFlagDetail(BaseModel):
    """Details of a detected Red Flag."""
    
    keyword: str
    category: str
    severity: float
    matched_text: str


class TriageResponse(BaseModel):
    """Response schema for triage predictions."""
    
    request_id: str
    timestamp: datetime
    
    # Acuity prediction
    acuity_level: int = Field(ge=1, le=5)
    acuity_label: str
    target_time_minutes: int
    
    # Confidence and probabilities
    confidence: float
    probabilities: Dict[str, float]
    
    # Red Flag information
    red_flag_override: bool
    red_flags: List[RedFlagDetail]
    
    # Critical vital findings
    critical_vitals: List[str]
    
    # Additional info
    recommendation: str
    requires_physician_review: bool
    
    # Audit trail
    audit_message: str


class OverrideRequest(BaseModel):
    """
    Request schema for clinician override.
    
    This endpoint allows clinicians to correct AI predictions,
    providing ground truth data for model retraining.
    """
    
    original_request_id: str = Field(
        ...,
        description="Request ID from the original triage prediction",
    )
    corrected_acuity: int = Field(
        ...,
        ge=1,
        le=5,
        description="Clinician's corrected acuity level (1-5)",
    )
    override_reason: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Clinical reasoning for the override",
    )
    clinician_id: str = Field(
        ...,
        min_length=1,
        description="ID of the overriding clinician",
    )
    clinician_role: Optional[str] = Field(
        default=None,
        description="Role of clinician (physician, nurse, pa, etc.)",
    )
    
    # Optional: Include updated clinical information
    additional_symptoms: Optional[str] = Field(
        default=None,
        description="Any additional symptoms discovered",
    )
    updated_vitals: Optional[VitalsInput] = Field(
        default=None,
        description="Updated vital signs if changed",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_request_id": "abc123-def456",
                "corrected_acuity": 2,
                "override_reason": "Patient has history of MI, ECG shows ST elevation. Escalating to Level 2 for immediate cardiology consult.",
                "clinician_id": "dr_smith_12345",
                "clinician_role": "physician",
                "additional_symptoms": "Diaphoresis noted on exam, patient appears anxious",
            }
        }


class OverrideResponse(BaseModel):
    """Response schema for override requests."""
    
    override_id: str
    original_request_id: str
    timestamp: datetime
    
    original_acuity: int
    corrected_acuity: int
    
    clinician_id: str
    override_reason: str
    
    status: str
    message: str
    
    # Flag for retraining
    queued_for_training: bool


class HealthResponse(BaseModel):
    """Health check response."""
    
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
    
    Holds the loaded model, override storage, and pending training data.
    """
    
    def __init__(self):
        self.classifier: Optional[TriageClassifier] = None
        self.parser: Optional[RedFlagParser] = None
        self.is_ready: bool = False
        
        # In-memory storage for triage history and overrides
        # In production, use database/Redis
        self.triage_history: Dict[str, Dict[str, Any]] = {}
        self.overrides: Dict[str, Dict[str, Any]] = {}
        self.pending_training_data: List[Dict[str, Any]] = []
        
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the classifier and parser."""
        async with self._lock:
            try:
                # Initialize Red Flag parser
                self.parser = get_parser()
                
                # Initialize and train classifier
                # In production, load from MLflow registry
                self.classifier = TriageClassifier()
                
                # Try to load pre-trained model
                try:
                    from model import create_sample_data
                    X, y = create_sample_data(n_samples=3000)
                    self.classifier.fit(X, y)
                    logger.info("Classifier trained on sample data")
                except Exception as e:
                    logger.warning(f"Could not train classifier: {e}")
                
                self.is_ready = True
                logger.info("Application state initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize: {e}")
                self.is_ready = False
    
    def store_triage(self, request_id: str, data: Dict[str, Any]) -> None:
        """Store triage request for audit and potential retraining."""
        self.triage_history[request_id] = {
            **data,
            "stored_at": datetime.utcnow().isoformat(),
        }
        
        # Limit in-memory storage
        if len(self.triage_history) > 10000:
            # Remove oldest entries
            oldest_keys = sorted(self.triage_history.keys())[:1000]
            for key in oldest_keys:
                del self.triage_history[key]
    
    def store_override(self, override_id: str, data: Dict[str, Any]) -> None:
        """Store override for retraining."""
        self.overrides[override_id] = {
            **data,
            "stored_at": datetime.utcnow().isoformat(),
        }
        
        # Add to pending training data
        self.pending_training_data.append(data)
    
    def get_pending_training_data(self) -> List[Dict[str, Any]]:
        """Get and clear pending training data."""
        data = self.pending_training_data.copy()
        self.pending_training_data = []
        return data


# Global state
app_state = AppState()


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    await app_state.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down triage agent")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Triage & Acuity Agent",
    description="""
    AI-powered Emergency Department triage system.
    
    ## Features
    - **Acuity Prediction**: Predicts ESI levels 1-5 based on symptoms and vitals
    - **Red Flag Detection**: Automatically escalates critical symptoms
    - **Clinician Override**: Allows corrections for continuous learning
    - **Federated Learning**: Privacy-preserving collaborative training
    
    ## Safety First
    This system includes multiple safety mechanisms:
    1. Red Flag keyword detection overrides ML predictions
    2. Critical vital signs trigger automatic escalation
    3. Low-confidence predictions are flagged for review
    4. All predictions are logged for clinical audit
    
    ## Acuity Levels
    - **Level 1 (Critical)**: Immediate life-saving intervention
    - **Level 2 (Emergent)**: High-risk, don't delay
    - **Level 3 (Urgent)**: Stable, needs multiple resources
    - **Level 4 (Less Urgent)**: Stable, single resource
    - **Level 5 (Non-Urgent)**: Could be seen in clinic
    """,
    version=settings.service_version,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def prepare_features(request: TriageRequest, red_flag_count: int) -> Dict[str, Any]:
    """
    Prepare feature dictionary from triage request.
    
    Args:
        request: The triage request
        red_flag_count: Number of Red Flags detected
    
    Returns:
        Dictionary of features for the classifier
    """
    features = {
        "red_flag_count": red_flag_count,
    }
    
    # Vitals
    if request.vitals:
        features.update({
            "heart_rate": request.vitals.heart_rate,
            "systolic_bp": request.vitals.systolic_bp,
            "diastolic_bp": request.vitals.diastolic_bp,
            "respiratory_rate": request.vitals.respiratory_rate,
            "oxygen_saturation": request.vitals.oxygen_saturation,
            "temperature": request.vitals.temperature,
            "gcs": request.vitals.gcs,
        })
    
    # Patient info
    if request.patient:
        features["age"] = request.patient.age
        features["gender"] = 1 if request.patient.gender == "male" else 0
        
        arrival_modes = {"walk_in": 0, "ambulance": 1, "helicopter": 2}
        features["arrival_mode"] = arrival_modes.get(request.patient.arrival_mode, 0)
    
    # Pain and complaint
    features["pain_score"] = request.pain_score
    
    if request.chief_complaint_category:
        from model import TriageClassifier
        features["chief_complaint_category"] = (
            TriageClassifier.COMPLAINT_CATEGORIES.get(
                request.chief_complaint_category, 20
            )
        )
    
    return features


def check_critical_vitals(vitals: Optional[VitalsInput]) -> List[str]:
    """Check for critically abnormal vital signs."""
    if not vitals:
        return []
    
    vs = VitalSigns(
        heart_rate=vitals.heart_rate,
        systolic_bp=vitals.systolic_bp,
        diastolic_bp=vitals.diastolic_bp,
        respiratory_rate=vitals.respiratory_rate,
        oxygen_saturation=vitals.oxygen_saturation,
        temperature=vitals.temperature,
        gcs=vitals.gcs,
    )
    
    return vs.check_critical_vitals()


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health_check() -> HealthResponse:
    """
    Health check endpoint for Kubernetes probes.
    
    Checks:
    - Service status
    - Model availability
    - Parser readiness
    """
    checks = {}
    overall_status = "healthy"
    
    # Model check
    model_check = {
        "status": "ok" if app_state.classifier and app_state.classifier.is_fitted else "degraded",
        "is_fitted": app_state.classifier.is_fitted if app_state.classifier else False,
    }
    if not app_state.classifier or not app_state.classifier.is_fitted:
        overall_status = "degraded"
    checks["model"] = model_check
    
    # Parser check
    parser_check = {
        "status": "ok" if app_state.parser else "error",
    }
    if not app_state.parser:
        overall_status = "unhealthy"
    checks["parser"] = parser_check
    
    # Pending overrides
    checks["overrides"] = {
        "status": "ok",
        "pending_training_samples": len(app_state.pending_training_data),
        "total_overrides": len(app_state.overrides),
    }
    
    return HealthResponse(
        status=overall_status,
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.utcnow(),
        checks=checks,
    )


@app.post("/triage", response_model=TriageResponse, tags=["Triage"])
async def triage_patient(request: TriageRequest) -> TriageResponse:
    """
    Perform AI-assisted triage on a patient.
    
    This endpoint:
    1. Scans symptoms for Red Flag keywords
    2. Checks vitals for critical abnormalities
    3. Runs XGBoost classifier for acuity prediction
    4. Applies safety overrides if needed
    5. Returns comprehensive triage recommendation
    
    **Red Flag Override:**
    If critical symptoms (e.g., "chest pain", "can't breathe") are detected,
    the system automatically escalates to Level 1 or 2 regardless of ML prediction.
    This is a critical safety mechanism for Emergency Medicine.
    
    **Example Request:**
    ```json
    {
        "symptoms": "58 year old male with chest pain and shortness of breath",
        "vitals": {"heart_rate": 102, "systolic_bp": 158, "oxygen_saturation": 94},
        "patient": {"age": 58, "gender": "male", "arrival_mode": "ambulance"}
    }
    ```
    """
    request_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    logger.info(
        f"Triage request: {request_id}",
        extra={"symptoms_length": len(request.symptoms)},
    )
    
    # -------------------------------------------------------------------------
    # Step 1: Red Flag Detection (Safety Override)
    # -------------------------------------------------------------------------
    parse_result = parse_symptoms(request.symptoms)
    red_flags = parse_result.red_flags
    
    red_flag_details = [
        RedFlagDetail(
            keyword=rf.matched_text,
            category=rf.category.value,
            severity=rf.severity,
            matched_text=rf.matched_text,
        )
        for rf in red_flags
    ]
    
    # -------------------------------------------------------------------------
    # Step 2: Check Critical Vitals
    # -------------------------------------------------------------------------
    critical_vitals = check_critical_vitals(request.vitals)
    
    # -------------------------------------------------------------------------
    # Step 3: ML Prediction
    # -------------------------------------------------------------------------
    ml_acuity = 3  # Default to Urgent
    confidence = 0.5
    probabilities = {f"level_{i}": 0.2 for i in range(1, 6)}
    
    if app_state.classifier and app_state.classifier.is_fitted:
        try:
            import pandas as pd
            
            features = prepare_features(request, len(red_flags))
            features_df = pd.DataFrame([features])
            
            predictions = app_state.classifier.predict_with_confidence(features_df)
            if predictions:
                pred = predictions[0]
                ml_acuity = pred["predicted_acuity"]
                confidence = pred["confidence"]
                probabilities = pred["probabilities"]
                
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
    
    # -------------------------------------------------------------------------
    # Step 4: Apply Safety Overrides
    # -------------------------------------------------------------------------
    final_acuity = ml_acuity
    red_flag_override = False
    
    # Red Flag override
    if red_flags and settings.red_flag_override_enabled:
        override_level = min(rf.suggested_acuity for rf in red_flags)
        if override_level < final_acuity:
            final_acuity = override_level
            red_flag_override = True
            logger.warning(
                f"RED FLAG OVERRIDE: {request_id}",
                extra={
                    "original_acuity": ml_acuity,
                    "override_acuity": final_acuity,
                    "red_flags": [rf.matched_text for rf in red_flags],
                }
            )
    
    # Critical vitals override
    if critical_vitals:
        if final_acuity > 2:  # Escalate to at least Level 2
            final_acuity = 2
            logger.warning(
                f"CRITICAL VITALS OVERRIDE: {request_id}",
                extra={"critical_findings": critical_vitals},
            )
    
    # -------------------------------------------------------------------------
    # Step 5: Generate Recommendation
    # -------------------------------------------------------------------------
    acuity_label = AcuityLevel.get_label(final_acuity)
    target_time = AcuityLevel.get_target_time(final_acuity)
    
    # Determine if physician review required
    requires_review = (
        confidence < 0.6 or
        final_acuity <= 2 or
        red_flag_override or
        len(critical_vitals) > 0
    )
    
    # Generate recommendation text
    if final_acuity == 1:
        recommendation = "IMMEDIATE: Patient requires immediate resuscitation. Notify physician NOW."
    elif final_acuity == 2:
        recommendation = "EMERGENT: High-risk patient. Physician evaluation within 10 minutes."
    elif final_acuity == 3:
        recommendation = "URGENT: Stable patient needing multiple resources. Target: 30 minutes."
    elif final_acuity == 4:
        recommendation = "LESS URGENT: Stable patient, single resource needed. Target: 60 minutes."
    else:
        recommendation = "NON-URGENT: Minor complaint, could be seen in clinic. Target: 120 minutes."
    
    # Audit message
    audit_parts = [f"Triage completed for {request_id}."]
    if red_flag_override:
        audit_parts.append(f"RED FLAG OVERRIDE applied: {[rf.matched_text for rf in red_flags]}")
    if critical_vitals:
        audit_parts.append(f"Critical vitals detected: {critical_vitals}")
    audit_parts.append(f"Final acuity: Level {final_acuity} ({acuity_label})")
    audit_message = " ".join(audit_parts)
    
    # -------------------------------------------------------------------------
    # Step 6: Store for Audit and Potential Retraining
    # -------------------------------------------------------------------------
    app_state.store_triage(request_id, {
        "request": request.dict(),
        "prediction": {
            "ml_acuity": ml_acuity,
            "final_acuity": final_acuity,
            "confidence": confidence,
            "red_flag_override": red_flag_override,
        },
        "timestamp": timestamp.isoformat(),
    })
    
    # -------------------------------------------------------------------------
    # Build Response
    # -------------------------------------------------------------------------
    response = TriageResponse(
        request_id=request_id,
        timestamp=timestamp,
        acuity_level=final_acuity,
        acuity_label=acuity_label,
        target_time_minutes=target_time,
        confidence=confidence,
        probabilities=probabilities,
        red_flag_override=red_flag_override,
        red_flags=red_flag_details,
        critical_vitals=critical_vitals,
        recommendation=recommendation,
        requires_physician_review=requires_review,
        audit_message=audit_message,
    )
    
    logger.info(
        f"Triage complete: {request_id}",
        extra={
            "acuity": final_acuity,
            "confidence": confidence,
            "red_flag_override": red_flag_override,
        }
    )
    
    return response


@app.post("/override", response_model=OverrideResponse, tags=["Triage"])
async def override_triage(request: OverrideRequest) -> OverrideResponse:
    """
    Submit a clinician override for an AI triage prediction.
    
    This endpoint is **CRUCIAL** for the Continuous Learning loop:
    
    1. **Immediate Effect**: Updates the patient's triage status
    2. **Training Data**: Override is stored as ground truth for retraining
    3. **Quality Improvement**: Helps identify model weaknesses
    4. **Audit Trail**: Documents clinical decision-making
    
    **Why Overrides Matter for Active Learning:**
    
    Traditional ML models are trained once and deployed. But medical AI
    needs continuous improvement because:
    - Patient populations change
    - New diseases emerge (e.g., COVID-19)
    - Clinical practices evolve
    - Edge cases are discovered
    
    Each override provides a labeled example where:
    - Input: Original patient data (symptoms, vitals)
    - Label: Clinician's corrected acuity (ground truth)
    
    These override samples are fed into the next Federated Learning
    training round, improving the model for all hospitals in the network.
    
    **Example Override:**
    ```json
    {
        "original_request_id": "abc123-def456",
        "corrected_acuity": 2,
        "override_reason": "ECG shows STEMI, escalating to emergent",
        "clinician_id": "dr_smith_12345",
        "clinician_role": "physician"
    }
    ```
    """
    override_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    
    logger.info(
        f"Override request: {override_id}",
        extra={
            "original_id": request.original_request_id,
            "clinician": request.clinician_id,
            "corrected_acuity": request.corrected_acuity,
        }
    )
    
    # -------------------------------------------------------------------------
    # Validate Original Request Exists
    # -------------------------------------------------------------------------
    original_data = app_state.triage_history.get(request.original_request_id)
    
    if not original_data:
        # In production, would check database
        logger.warning(f"Original request not found: {request.original_request_id}")
        # Continue anyway - we still want to record the override
        original_acuity = 3  # Unknown
    else:
        original_acuity = original_data["prediction"]["final_acuity"]
    
    # -------------------------------------------------------------------------
    # Store Override for Training
    # -------------------------------------------------------------------------
    override_data = {
        "override_id": override_id,
        "original_request_id": request.original_request_id,
        "original_acuity": original_acuity,
        "corrected_acuity": request.corrected_acuity,
        "override_reason": request.override_reason,
        "clinician_id": request.clinician_id,
        "clinician_role": request.clinician_role,
        "additional_symptoms": request.additional_symptoms,
        "updated_vitals": request.updated_vitals.dict() if request.updated_vitals else None,
        "timestamp": timestamp.isoformat(),
    }
    
    # If we have the original data, create training sample
    if original_data:
        training_sample = {
            "features": original_data["request"],
            "original_label": original_acuity,
            "corrected_label": request.corrected_acuity,
            "override_reason": request.override_reason,
            "timestamp": timestamp.isoformat(),
        }
        override_data["training_sample"] = training_sample
    
    app_state.store_override(override_id, override_data)
    
    # -------------------------------------------------------------------------
    # Log for Audit
    # -------------------------------------------------------------------------
    logger.warning(
        "TRIAGE OVERRIDE RECORDED",
        extra={
            "override_id": override_id,
            "original_request_id": request.original_request_id,
            "original_acuity": original_acuity,
            "corrected_acuity": request.corrected_acuity,
            "clinician_id": request.clinician_id,
            "reason": request.override_reason[:100],
        }
    )
    
    # -------------------------------------------------------------------------
    # Build Response
    # -------------------------------------------------------------------------
    acuity_change = "escalated" if request.corrected_acuity < original_acuity else "de-escalated"
    
    response = OverrideResponse(
        override_id=override_id,
        original_request_id=request.original_request_id,
        timestamp=timestamp,
        original_acuity=original_acuity,
        corrected_acuity=request.corrected_acuity,
        clinician_id=request.clinician_id,
        override_reason=request.override_reason,
        status="accepted",
        message=f"Override recorded. Patient {acuity_change} from Level {original_acuity} to Level {request.corrected_acuity}. Data queued for model retraining.",
        queued_for_training=True,
    )
    
    return response


@app.get("/overrides/pending", tags=["Training"])
async def get_pending_overrides() -> Dict[str, Any]:
    """
    Get pending override data for federated learning.
    
    This endpoint is called by the FL training system to collect
    new labeled data from clinician overrides.
    """
    pending = app_state.pending_training_data
    
    return {
        "count": len(pending),
        "samples": pending[:100],  # Limit response size
        "message": f"Retrieved {len(pending)} pending training samples",
    }


@app.get("/stats", tags=["Operations"])
async def get_stats() -> Dict[str, Any]:
    """Get triage service statistics."""
    
    # Calculate override rate
    total_triages = len(app_state.triage_history)
    total_overrides = len(app_state.overrides)
    
    override_rate = (total_overrides / total_triages * 100) if total_triages > 0 else 0
    
    return {
        "total_triages": total_triages,
        "total_overrides": total_overrides,
        "override_rate_percent": round(override_rate, 2),
        "pending_training_samples": len(app_state.pending_training_data),
        "model_loaded": app_state.classifier is not None and app_state.classifier.is_fitted,
        "red_flag_override_enabled": settings.red_flag_override_enabled,
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
