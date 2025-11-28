"""
FastAPI application for ER/OR Scheduling Agent.

Provides REST endpoints for:
- ER queue management (reordering based on triage scores)
- OR scheduling (optimizing surgery assignments)

This is the interface layer for the Hybrid ML + Optimization Agent.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import config, PROCEDURE_TYPES, OR_ROOMS
from duration_predictor import (
    DurationPredictor,
    PredictionInput,
    get_or_create_predictor
)
from scheduler import (
    ORScheduler,
    ERQueueManager,
    Surgery,
    ORBlock,
    SchedulingStatus
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
duration_predictor: Optional[DurationPredictor] = None
or_scheduler: Optional[ORScheduler] = None
er_queue_manager: Optional[ERQueueManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler - initialize models on startup."""
    global duration_predictor, or_scheduler, er_queue_manager
    
    logger.info("Initializing ER/OR Scheduling Agent...")
    
    # Initialize duration predictor (trains model if needed)
    duration_predictor = get_or_create_predictor()
    
    # Initialize scheduler with predictor
    or_scheduler = ORScheduler(duration_predictor)
    
    # Initialize ER queue manager
    er_queue_manager = ERQueueManager()
    
    logger.info("ER/OR Scheduling Agent ready")
    
    yield
    
    logger.info("Shutting down ER/OR Scheduling Agent")


# Create FastAPI app
app = FastAPI(
    title="ER/OR Scheduling Agent",
    description="""
    Hybrid Agent combining ML-based surgery duration prediction with 
    constraint-based OR scheduling optimization.
    
    ## Features
    - **ER Queue Management**: Reorder waiting lists based on triage scores
    - **OR Scheduling**: Optimize surgery assignments to minimize idle time
    - **Duration Prediction**: ML-powered surgery duration estimation
    
    ## Architecture
    This agent demonstrates the Hybrid Agent pattern:
    - ML Component: XGBoost model predicts surgery durations
    - Optimization Component: OR-Tools solves bin packing for scheduling
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models
# ============================================================================

class PatientQueueItem(BaseModel):
    """Patient in ER queue."""
    patient_id: str
    name: Optional[str] = None
    acuity: int = Field(ge=1, le=5, description="ESI acuity level (1=most urgent)")
    triage_score: float = Field(ge=0, le=1, description="Triage agent score")
    wait_time_minutes: int = Field(ge=0, default=0)
    chief_complaint: Optional[str] = None
    is_pediatric: bool = False
    is_geriatric: bool = False


class ReorderQueueRequest(BaseModel):
    """Request to reorder ER queue."""
    patients: List[PatientQueueItem]


class ReorderQueueResponse(BaseModel):
    """Response with reordered queue."""
    success: bool
    queue: List[Dict[str, Any]]
    total_patients: int
    critical_count: int
    average_wait_minutes: float


class SurgeryRequest(BaseModel):
    """Surgery scheduling request."""
    surgery_id: str
    patient_id: str
    procedure_type: str
    patient_acuity: int = Field(ge=1, le=5, default=3)
    patient_age: int = Field(ge=0, le=120, default=50)
    patient_bmi: float = Field(ge=10, le=60, default=25.0)
    comorbidity_count: int = Field(ge=0, le=10, default=0)
    surgeon_id: Optional[str] = None
    surgeon_experience_years: int = Field(ge=0, le=50, default=10)
    is_emergency: bool = False
    required_equipment: List[str] = []
    specialty: str = "general"


class ORBlockConfig(BaseModel):
    """OR block configuration."""
    block_id: str
    room_id: str
    start_time: datetime
    end_time: datetime
    specialty: List[str] = []
    equipment: List[str] = []


class ScheduleRequest(BaseModel):
    """Request to schedule surgeries."""
    surgeries: List[SurgeryRequest]
    start_date: Optional[datetime] = None
    num_days: int = Field(ge=1, le=7, default=1)
    or_blocks: Optional[List[ORBlockConfig]] = None


class ScheduledSurgeryResponse(BaseModel):
    """Scheduled surgery in response."""
    surgery_id: str
    patient_id: str
    procedure_type: str
    block_id: str
    room_id: str
    scheduled_start: datetime
    scheduled_end: datetime
    predicted_duration_minutes: int
    priority_score: float


class ScheduleResponse(BaseModel):
    """Response with optimized schedule."""
    success: bool
    status: str
    scheduled_surgeries: List[ScheduledSurgeryResponse]
    unscheduled_surgery_ids: List[str]
    total_scheduled: int
    total_unscheduled: int
    utilization_percent: float
    total_idle_minutes: int
    solver_time_seconds: float


class DurationPredictionRequest(BaseModel):
    """Request for surgery duration prediction."""
    procedure_type: str
    patient_acuity: int = Field(ge=1, le=5, default=3)
    patient_age: int = Field(ge=0, le=120, default=50)
    patient_bmi: float = Field(ge=10, le=60, default=25.0)
    comorbidity_count: int = Field(ge=0, le=10, default=0)
    surgeon_experience_years: int = Field(ge=0, le=50, default=10)
    is_emergency: bool = False


class DurationPredictionResponse(BaseModel):
    """Response with duration prediction."""
    predicted_duration_minutes: int
    confidence_interval_lower: int
    confidence_interval_upper: int
    procedure_type: str
    model_type: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    agent: str
    version: str
    model_loaded: bool
    uptime_info: Dict[str, Any]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        agent="eror_scheduling",
        version="1.0.0",
        model_loaded=duration_predictor is not None and duration_predictor.model is not None,
        uptime_info={
            "scheduler_ready": or_scheduler is not None,
            "queue_manager_ready": er_queue_manager is not None,
            "procedure_types_count": len(PROCEDURE_TYPES),
            "or_rooms_count": len(OR_ROOMS)
        }
    )


@app.post("/er/reorder-queue", response_model=ReorderQueueResponse)
async def reorder_er_queue(request: ReorderQueueRequest):
    """
    Reorder ER waiting queue based on triage scores and acuity.
    
    This endpoint takes a list of patients in the ER queue and returns
    them sorted by priority, considering:
    - ESI acuity level (1-5, where 1 is most critical)
    - Triage agent score (0-1)
    - Wait time (longer wait = higher priority)
    - Special populations (pediatric, geriatric)
    
    **Use Case**: Called after triage assessment to update queue order.
    """
    if er_queue_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue manager not initialized"
        )
    
    try:
        # Convert to dict format for queue manager
        patients_data = [p.model_dump() for p in request.patients]
        
        # Reorder queue
        reordered = er_queue_manager.reorder_queue(patients_data)
        
        # Calculate statistics
        critical_count = sum(1 for p in reordered if p.get("acuity", 3) <= 2)
        avg_wait = (
            sum(p.get("wait_time_minutes", 0) for p in reordered) / len(reordered)
            if reordered else 0
        )
        
        logger.info(f"Reordered ER queue: {len(reordered)} patients")
        
        return ReorderQueueResponse(
            success=True,
            queue=reordered,
            total_patients=len(reordered),
            critical_count=critical_count,
            average_wait_minutes=round(avg_wait, 1)
        )
    
    except Exception as e:
        logger.error(f"Error reordering queue: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reorder queue: {str(e)}"
        )


@app.post("/or/schedule", response_model=ScheduleResponse)
async def schedule_surgeries(request: ScheduleRequest):
    """
    Schedule surgeries to OR blocks using optimization.
    
    This endpoint takes a list of surgeries and returns an optimized
    schedule that:
    - Minimizes OR idle time (bin packing optimization)
    - Prioritizes high-acuity and emergency cases
    - Respects equipment and specialty constraints
    - Uses ML to predict surgery durations
    
    **Algorithm**: Constraint Programming (OR-Tools) with bin packing formulation.
    
    **Use Case**: Daily OR scheduling, emergency surgery insertion.
    """
    if or_scheduler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not initialized"
        )
    
    try:
        # Convert requests to Surgery objects
        surgeries = [
            Surgery(
                surgery_id=s.surgery_id,
                patient_id=s.patient_id,
                procedure_type=s.procedure_type,
                patient_acuity=s.patient_acuity,
                patient_age=s.patient_age,
                patient_bmi=s.patient_bmi,
                comorbidity_count=s.comorbidity_count,
                surgeon_id=s.surgeon_id,
                surgeon_experience_years=s.surgeon_experience_years,
                is_emergency=s.is_emergency,
                required_equipment=s.required_equipment,
                specialty=s.specialty
            )
            for s in request.surgeries
        ]
        
        # Convert OR blocks if provided
        or_blocks = None
        if request.or_blocks:
            or_blocks = [
                ORBlock(
                    block_id=b.block_id,
                    room_id=b.room_id,
                    start_time=b.start_time,
                    end_time=b.end_time,
                    capacity_minutes=int((b.end_time - b.start_time).total_seconds() / 60),
                    specialty=b.specialty,
                    equipment=b.equipment
                )
                for b in request.or_blocks
            ]
        
        # Run scheduling optimization
        result = or_scheduler.schedule(
            surgeries=surgeries,
            start_date=request.start_date,
            num_days=request.num_days,
            or_blocks=or_blocks
        )
        
        # Build response
        scheduled_responses = [
            ScheduledSurgeryResponse(
                surgery_id=s.surgery.surgery_id,
                patient_id=s.surgery.patient_id,
                procedure_type=s.surgery.procedure_type,
                block_id=s.block_id,
                room_id=s.room_id,
                scheduled_start=s.scheduled_start,
                scheduled_end=s.scheduled_end,
                predicted_duration_minutes=s.surgery.predicted_duration,
                priority_score=s.surgery.priority_score
            )
            for s in result.scheduled_surgeries
        ]
        
        unscheduled_ids = [s.surgery_id for s in result.unscheduled_surgeries]
        
        logger.info(
            f"Scheduled {len(scheduled_responses)} surgeries, "
            f"{len(unscheduled_ids)} unscheduled"
        )
        
        return ScheduleResponse(
            success=result.status in [SchedulingStatus.OPTIMAL, SchedulingStatus.FEASIBLE],
            status=result.status.value,
            scheduled_surgeries=scheduled_responses,
            unscheduled_surgery_ids=unscheduled_ids,
            total_scheduled=len(scheduled_responses),
            total_unscheduled=len(unscheduled_ids),
            utilization_percent=result.utilization_percent,
            total_idle_minutes=result.total_idle_minutes,
            solver_time_seconds=result.solver_time_seconds
        )
    
    except Exception as e:
        logger.error(f"Error scheduling surgeries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to schedule surgeries: {str(e)}"
        )


@app.post("/predict/duration", response_model=DurationPredictionResponse)
async def predict_surgery_duration(request: DurationPredictionRequest):
    """
    Predict surgery duration using ML model.
    
    This endpoint exposes the duration prediction model directly,
    useful for:
    - Pre-operative planning
    - Resource allocation
    - Patient communication
    
    **Model**: XGBoost regressor trained on historical surgery data.
    """
    if duration_predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Duration predictor not initialized"
        )
    
    try:
        pred_input = PredictionInput(
            procedure_type=request.procedure_type,
            patient_acuity=request.patient_acuity,
            patient_age=request.patient_age,
            patient_bmi=request.patient_bmi,
            comorbidity_count=request.comorbidity_count,
            surgeon_experience_years=request.surgeon_experience_years,
            is_emergency=request.is_emergency
        )
        
        result = duration_predictor.predict(pred_input)
        
        model_type = "xgboost" if duration_predictor.model is not None else "heuristic"
        
        return DurationPredictionResponse(
            predicted_duration_minutes=result.predicted_duration_minutes,
            confidence_interval_lower=result.confidence_interval_lower,
            confidence_interval_upper=result.confidence_interval_upper,
            procedure_type=request.procedure_type,
            model_type=model_type
        )
    
    except Exception as e:
        logger.error(f"Error predicting duration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to predict duration: {str(e)}"
        )


@app.get("/config/procedure-types")
async def get_procedure_types():
    """Get available procedure types and their base configurations."""
    return {
        "procedure_types": PROCEDURE_TYPES,
        "count": len(PROCEDURE_TYPES)
    }


@app.get("/config/or-rooms")
async def get_or_rooms():
    """Get available OR room configurations."""
    return {
        "or_rooms": OR_ROOMS,
        "count": len(OR_ROOMS)
    }


@app.get("/config/acuity-weights")
async def get_acuity_weights():
    """Get acuity level weights used for prioritization."""
    return {
        "weights": config.acuity.weights,
        "esi_mapping": config.acuity.esi_mapping
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=config.server.host,
        port=config.server.port,
        reload=config.server.debug
    )
