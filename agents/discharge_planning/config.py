"""
Discharge Planning Agent - Configuration Module

This module centralizes all environment-based configuration for the discharge
planning microservice. It follows the 12-factor app methodology by externalizing
configuration through environment variables.

================================================================================
AGENT PURPOSE & CLINICAL CONTEXT
================================================================================

The Discharge Planning Agent serves a critical role in hospital operations:

1. BED TURNOVER OPTIMIZATION:
   - Hospital beds are expensive resources (~$2,000-$3,000/day)
   - Each hour a patient stays beyond medical necessity costs money
   - This agent identifies patients who are clinically ready for discharge

2. PATIENT SAFETY BALANCE:
   - Premature discharge leads to readmissions (costly and dangerous)
   - 30-day readmission rates are tracked by CMS and affect reimbursement
   - The agent must balance efficiency with patient safety

3. CLINICAL WORKFLOW INTEGRATION:
   - Case managers and charge nurses review discharge candidates daily
   - This agent provides decision support, NOT autonomous decisions
   - Final discharge authority always rests with the attending physician

================================================================================
"""

from pydantic_settings import BaseSettings
from pydantic import Field, PostgresDsn
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Configuration is organized into logical sections that map to
    different aspects of the discharge planning workflow.
    """
    
    # ==========================================================================
    # SERVICE IDENTIFICATION
    # ==========================================================================
    service_name: str = Field(
        default="discharge-planning-agent",
        description="Unique identifier for this microservice"
    )
    service_version: str = Field(
        default="1.0.0",
        description="Semantic version of this agent"
    )
    environment: str = Field(
        default="development",
        description="Runtime environment (development, staging, production)"
    )
    
    # ==========================================================================
    # DATABASE CONFIGURATION
    # ==========================================================================
    # PostgreSQL connection for fetching patient clinical data
    database_url: PostgresDsn = Field(
        default="postgresql://postgres:postgres@postgres:5432/hospital_ai",
        description="PostgreSQL connection string for patient data"
    )
    db_pool_size: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Database connection pool size"
    )
    db_pool_timeout: int = Field(
        default=30,
        description="Timeout in seconds for acquiring a connection from pool"
    )
    
    # ==========================================================================
    # MODEL CONFIGURATION
    # ==========================================================================
    # XGBoost model hyperparameters for readiness scoring
    xgb_n_estimators: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Number of gradient boosted trees"
    )
    xgb_max_depth: int = Field(
        default=6,
        ge=2,
        le=15,
        description="Maximum tree depth (controls model complexity)"
    )
    xgb_learning_rate: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Boosting learning rate (shrinkage)"
    )
    xgb_min_child_weight: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum sum of instance weight in a child node"
    )
    
    # Model file paths
    model_path: str = Field(
        default="/app/models/discharge_readiness.json",
        description="Path to saved XGBoost model"
    )
    
    # ==========================================================================
    # CLINICAL RULE ENGINE CONFIGURATION
    # ==========================================================================
    # These thresholds define the clinical guardrails that can VETO discharge
    # even when the ML model predicts high readiness
    
    # Vital sign thresholds
    fever_threshold_celsius: float = Field(
        default=38.0,
        description="Temperature above which patient has fever (Celsius)"
    )
    tachycardia_threshold: int = Field(
        default=100,
        description="Heart rate above which indicates tachycardia"
    )
    bradycardia_threshold: int = Field(
        default=50,
        description="Heart rate below which indicates bradycardia"
    )
    hypoxia_threshold: float = Field(
        default=92.0,
        description="SpO2 below which indicates hypoxia (%)"
    )
    hypotension_systolic_threshold: int = Field(
        default=90,
        description="Systolic BP below which indicates hypotension"
    )
    hypertension_systolic_threshold: int = Field(
        default=180,
        description="Systolic BP above which indicates hypertensive crisis"
    )
    
    # Lab value thresholds
    wbc_high_threshold: float = Field(
        default=11.0,
        description="WBC count above which indicates possible infection (x10^9/L)"
    )
    wbc_low_threshold: float = Field(
        default=4.0,
        description="WBC count below which indicates leukopenia (x10^9/L)"
    )
    hemoglobin_low_threshold: float = Field(
        default=7.0,
        description="Hemoglobin below which requires transfusion consideration (g/dL)"
    )
    creatinine_high_threshold: float = Field(
        default=2.0,
        description="Creatinine above which indicates kidney impairment (mg/dL)"
    )
    potassium_low_threshold: float = Field(
        default=3.0,
        description="Potassium below which is dangerous (mEq/L)"
    )
    potassium_high_threshold: float = Field(
        default=5.5,
        description="Potassium above which is dangerous (mEq/L)"
    )
    
    # Readiness score thresholds
    readiness_score_high_threshold: float = Field(
        default=0.7,
        description="Score above which patient is 'likely ready' for discharge"
    )
    readiness_score_medium_threshold: float = Field(
        default=0.4,
        description="Score above which patient is 'possibly ready'"
    )
    
    # ==========================================================================
    # OPERATIONAL RULES
    # ==========================================================================
    # Minimum length of stay requirements by admission type
    min_los_hours_emergency: int = Field(
        default=24,
        description="Minimum hours before ER admission can be discharged"
    )
    min_los_hours_surgical: int = Field(
        default=48,
        description="Minimum hours after surgery before discharge"
    )
    min_los_hours_observation: int = Field(
        default=6,
        description="Minimum hours for observation status"
    )
    
    # ==========================================================================
    # API CONFIGURATION
    # ==========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8004,
        description="API server port"
    )
    api_workers: int = Field(
        default=1,
        description="Number of Uvicorn workers"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    
    # ==========================================================================
    # ORCHESTRATOR INTEGRATION
    # ==========================================================================
    orchestrator_url: str = Field(
        default="http://orchestrator:3000",
        description="URL of the central orchestrator service"
    )
    orchestrator_callback_enabled: bool = Field(
        default=True,
        description="Whether to notify orchestrator of discharge candidates"
    )
    
    # ==========================================================================
    # LOGGING CONFIGURATION
    # ==========================================================================
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: str = Field(
        default="json",
        description="Log format (json, text)"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    
    Using lru_cache ensures we only parse environment variables once,
    improving performance and consistency across the application.
    
    Returns:
        Settings: Application configuration object
    """
    return Settings()


# ==========================================================================
# CONVENIENCE EXPORTS
# ==========================================================================
settings = get_settings()
