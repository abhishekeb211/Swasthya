"""
Demand Forecast Agent - FastAPI Application

This module provides the REST API for the demand forecasting service.
It exposes endpoints for predictions, training triggers, and health checks.

================================================================================
API DESIGN PHILOSOPHY
================================================================================

This API follows RESTful principles adapted for ML inference:

1. STATELESS PREDICTIONS:
   - Each /predict call is independent
   - Model is loaded from registry, not kept in memory permanently
   - Enables horizontal scaling

2. ASYNC TRAINING:
   - /train triggers background training
   - Returns immediately with job ID
   - Polling or webhooks for completion

3. HEALTH CHECKS:
   - /health for Kubernetes probes
   - Includes model availability status
   - Database connectivity check

================================================================================
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from config import settings
from model import DemandForecaster
from mlflow_tracking import MLflowTracker
from train import DataLoader, train_model

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS (Request/Response Schemas)
# =============================================================================

class PredictionRequest(BaseModel):
    """Request schema for demand predictions."""
    
    horizon_hours: int = Field(
        default=168,
        ge=1,
        le=720,  # Max 30 days
        description="Number of hours to forecast",
    )
    start_datetime: Optional[datetime] = Field(
        default=None,
        description="Start of forecast period (default: now)",
    )
    include_components: bool = Field(
        default=False,
        description="Include trend/seasonality decomposition",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "horizon_hours": 168,
                "start_datetime": "2024-01-15T00:00:00Z",
                "include_components": False,
            }
        }


class PredictionPoint(BaseModel):
    """Single prediction point in the forecast."""
    
    timestamp: datetime
    predicted_count: int = Field(ge=0)
    lower_bound: int = Field(ge=0)
    upper_bound: int = Field(ge=0)


class PredictionResponse(BaseModel):
    """Response schema for demand predictions."""
    
    request_id: str
    model_version: Optional[str]
    generated_at: datetime
    horizon_hours: int
    predictions: List[PredictionPoint]
    components: Optional[Dict[str, List[float]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TrainingRequest(BaseModel):
    """Request schema for triggering model training."""
    
    use_synthetic_data: bool = Field(
        default=False,
        description="Use synthetic data instead of database",
    )
    test_days: int = Field(
        default=30,
        ge=7,
        le=90,
        description="Days to hold out for testing",
    )
    register_model: bool = Field(
        default=True,
        description="Register trained model in MLflow registry",
    )
    department: Optional[str] = Field(
        default=None,
        description="Filter training data by department",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "use_synthetic_data": False,
                "test_days": 30,
                "register_model": True,
                "department": None,
            }
        }


class TrainingResponse(BaseModel):
    """Response schema for training requests."""
    
    job_id: str
    status: str
    message: str
    started_at: datetime
    mlflow_run_id: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


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
    
    Holds cached model and tracking of background jobs.
    In production, consider using Redis for distributed state.
    """
    
    def __init__(self):
        self.forecaster: Optional[DemandForecaster] = None
        self.model_version: Optional[str] = None
        self.model_loaded_at: Optional[datetime] = None
        self.training_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get_forecaster(self) -> DemandForecaster:
        """Get or load the forecaster model."""
        async with self._lock:
            if self.forecaster is None:
                await self._load_model()
            return self.forecaster
    
    async def _load_model(self) -> None:
        """Load model from MLflow registry."""
        try:
            tracker = MLflowTracker()
            model_version = tracker.get_latest_model_version(
                settings.mlflow_model_registry_name,
                stage="Production",
            )
            
            if model_version:
                self.forecaster = tracker.load_model(
                    settings.mlflow_model_registry_name,
                    stage="Production",
                )
                self.model_version = f"v{model_version.version}"
                self.model_loaded_at = datetime.utcnow()
                logger.info(f"Loaded production model: {self.model_version}")
            else:
                # No production model, try staging
                staging_version = tracker.get_latest_model_version(
                    settings.mlflow_model_registry_name,
                    stage="Staging",
                )
                if staging_version:
                    self.forecaster = tracker.load_model(
                        settings.mlflow_model_registry_name,
                        stage="Staging",
                    )
                    self.model_version = f"v{staging_version.version}-staging"
                    self.model_loaded_at = datetime.utcnow()
                    logger.info(f"Loaded staging model: {self.model_version}")
                else:
                    logger.warning("No model available in registry")
                    
        except Exception as e:
            logger.error(f"Failed to load model from registry: {e}")
            # Create untrained forecaster as placeholder
            self.forecaster = None
    
    async def refresh_model(self) -> None:
        """Force reload model from registry."""
        async with self._lock:
            self.forecaster = None
            await self._load_model()


# Global application state
app_state = AppState()


# =============================================================================
# LIFESPAN MANAGEMENT
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(f"Starting {settings.service_name} v{settings.service_version}")
    
    # Try to load model at startup
    try:
        await app_state.get_forecaster()
    except Exception as e:
        logger.warning(f"Could not load model at startup: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down demand forecast agent")


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="Demand Forecast Agent",
    description="""
    Hospital patient volume forecasting service using Facebook Prophet.
    
    ## Features
    - **Hourly Predictions**: Forecast ER arrivals, admissions
    - **Confidence Intervals**: 80% prediction bounds
    - **Seasonality Awareness**: Weekly, daily, holiday patterns
    - **MLflow Integration**: Model versioning and tracking
    - **Federated Learning**: Privacy-preserving collaborative training
    
    ## Usage
    1. POST to `/predict` with forecast horizon
    2. GET `/health` for service status
    3. POST to `/train` to trigger retraining
    """,
    version=settings.service_version,
    lifespan=lifespan,
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    - Database connectivity
    """
    checks = {}
    overall_status = "healthy"
    
    # Check model availability
    model_check = {
        "status": "ok",
        "model_version": app_state.model_version,
        "loaded_at": app_state.model_loaded_at.isoformat() if app_state.model_loaded_at else None,
    }
    if app_state.forecaster is None:
        model_check["status"] = "degraded"
        model_check["message"] = "No model loaded"
        overall_status = "degraded"
    checks["model"] = model_check
    
    # Check database connectivity
    db_check = {"status": "ok"}
    try:
        loader = DataLoader()
        # Simple connectivity test
        with loader.engine.connect() as conn:
            conn.execute("SELECT 1")
        loader.close()
    except Exception as e:
        db_check["status"] = "error"
        db_check["message"] = str(e)
        overall_status = "unhealthy"
    checks["database"] = db_check
    
    # Check MLflow connectivity
    mlflow_check = {"status": "ok"}
    try:
        tracker = MLflowTracker()
        # Just initializing is enough to verify connectivity
        mlflow_check["tracking_uri"] = settings.mlflow_tracking_uri
    except Exception as e:
        mlflow_check["status"] = "warning"
        mlflow_check["message"] = str(e)
    checks["mlflow"] = mlflow_check
    
    return HealthResponse(
        status=overall_status,
        service=settings.service_name,
        version=settings.service_version,
        timestamp=datetime.utcnow(),
        checks=checks,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_demand(request: PredictionRequest) -> PredictionResponse:
    """
    Generate patient volume forecasts.
    
    Returns hourly predictions with confidence intervals for the specified
    forecast horizon. Predictions include:
    - Point estimates (expected patient count)
    - 80% confidence bounds
    - Optional seasonality decomposition
    
    **Example Use Cases:**
    - Staffing optimization: Plan nurse schedules 7 days ahead
    - Capacity planning: Predict bed utilization
    - Resource allocation: Anticipate equipment needs
    """
    request_id = str(uuid.uuid4())
    logger.info(
        f"Prediction request: {request_id}",
        extra={"horizon": request.horizon_hours},
    )
    
    # Get forecaster
    forecaster = await app_state.get_forecaster()
    
    if forecaster is None or not forecaster.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_not_available",
                "message": "No trained model available. Please train a model first.",
                "request_id": request_id,
            },
        )
    
    try:
        # Generate predictions
        predictions_df = forecaster.predict(
            horizon_hours=request.horizon_hours,
            start_date=request.start_datetime,
        )
        
        # Convert to response format
        predictions = [
            PredictionPoint(
                timestamp=row["ds"],
                predicted_count=row["yhat"],
                lower_bound=row["yhat_lower"],
                upper_bound=row["yhat_upper"],
            )
            for _, row in predictions_df.iterrows()
        ]
        
        # Optionally include components
        components = None
        if request.include_components:
            try:
                comp_df = forecaster.get_components()
                components = {
                    col: comp_df[col].tolist()
                    for col in comp_df.columns
                    if col != "ds"
                }
            except Exception as e:
                logger.warning(f"Could not extract components: {e}")
        
        response = PredictionResponse(
            request_id=request_id,
            model_version=app_state.model_version,
            generated_at=datetime.utcnow(),
            horizon_hours=request.horizon_hours,
            predictions=predictions,
            components=components,
            metadata={
                "model_params": forecaster.get_params(),
                "training_metadata": forecaster.training_metadata,
            },
        )
        
        logger.info(
            f"Prediction complete: {request_id}",
            extra={"n_predictions": len(predictions)},
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "prediction_failed",
                "message": str(e),
                "request_id": request_id,
            },
        )


@app.post("/train", response_model=TrainingResponse, tags=["Training"])
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
) -> TrainingResponse:
    """
    Trigger model retraining.
    
    This endpoint starts a background training job and returns immediately.
    The training process:
    1. Loads data from PostgreSQL (or generates synthetic)
    2. Splits into train/test sets
    3. Trains Prophet model
    4. Evaluates on test set
    5. Logs to MLflow and optionally registers model
    
    **Note:** For production, consider using a job queue (Celery, etc.)
    instead of FastAPI background tasks.
    """
    job_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    
    logger.info(
        f"Training job submitted: {job_id}",
        extra={
            "synthetic": request.use_synthetic_data,
            "department": request.department,
        },
    )
    
    # Initialize job tracking
    app_state.training_jobs[job_id] = {
        "status": "pending",
        "started_at": started_at,
        "completed_at": None,
        "mlflow_run_id": None,
        "metrics": None,
        "error": None,
    }
    
    async def run_training():
        """Background training task."""
        try:
            app_state.training_jobs[job_id]["status"] = "running"
            
            # Run training (synchronous, in thread pool)
            loop = asyncio.get_event_loop()
            model, metrics, run_id = await loop.run_in_executor(
                None,
                lambda: train_model(
                    use_synthetic=request.use_synthetic_data,
                    test_days=request.test_days,
                    register_model=request.register_model,
                    department=request.department,
                ),
            )
            
            app_state.training_jobs[job_id].update({
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "mlflow_run_id": run_id,
                "metrics": metrics,
            })
            
            # Refresh the model in memory
            await app_state.refresh_model()
            
            logger.info(f"Training job completed: {job_id}")
            
        except Exception as e:
            logger.error(f"Training job failed: {job_id} - {e}")
            app_state.training_jobs[job_id].update({
                "status": "failed",
                "completed_at": datetime.utcnow(),
                "error": str(e),
            })
    
    # Schedule background task
    background_tasks.add_task(run_training)
    
    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message="Training job submitted. Check /train/{job_id} for status.",
        started_at=started_at,
    )


@app.get("/train/{job_id}", response_model=TrainingResponse, tags=["Training"])
async def get_training_status(job_id: str) -> TrainingResponse:
    """
    Get status of a training job.
    
    Poll this endpoint to check if training has completed.
    """
    if job_id not in app_state.training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {job_id}",
        )
    
    job = app_state.training_jobs[job_id]
    
    return TrainingResponse(
        job_id=job_id,
        status=job["status"],
        message=job.get("error", f"Training {job['status']}"),
        started_at=job["started_at"],
        mlflow_run_id=job.get("mlflow_run_id"),
        metrics=job.get("metrics"),
    )


@app.post("/model/refresh", tags=["Operations"])
async def refresh_model() -> Dict[str, str]:
    """
    Force refresh the model from MLflow registry.
    
    Use this after manually promoting a model to production.
    """
    await app_state.refresh_model()
    
    return {
        "status": "success",
        "model_version": app_state.model_version or "none",
        "message": "Model refreshed from registry",
    }


@app.get("/model/info", tags=["Operations"])
async def get_model_info() -> Dict[str, Any]:
    """
    Get information about the currently loaded model.
    """
    forecaster = await app_state.get_forecaster()
    
    if forecaster is None:
        return {
            "loaded": False,
            "message": "No model currently loaded",
        }
    
    return {
        "loaded": True,
        "version": app_state.model_version,
        "loaded_at": app_state.model_loaded_at.isoformat() if app_state.model_loaded_at else None,
        "is_fitted": forecaster.is_fitted,
        "parameters": forecaster.get_params(),
        "training_metadata": forecaster.training_metadata,
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
