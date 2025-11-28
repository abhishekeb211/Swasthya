"""
Triage & Acuity Agent - Configuration Module

This module centralizes all environment-based configuration for the triage
microservice. It follows the 12-factor app methodology by externalizing
configuration through environment variables.

================================================================================
EMERGENCY SEVERITY INDEX (ESI) OVERVIEW
================================================================================

The ESI is a 5-level triage algorithm used in Emergency Departments:

    Level 1 - RESUSCITATION (Critical)
    ───────────────────────────────────
    Immediate life-threatening conditions requiring immediate physician attention.
    Examples: Cardiac arrest, severe respiratory distress, active seizure
    Target: 0 minutes wait time

    Level 2 - EMERGENT (High Risk)
    ──────────────────────────────
    High-risk situations, confused/lethargic/disoriented, severe pain/distress.
    Examples: Chest pain, stroke symptoms, severe allergic reaction
    Target: <10 minutes to physician

    Level 3 - URGENT (Moderate)
    ───────────────────────────
    Stable but needs multiple resources (labs, imaging, IV, etc.)
    Examples: Abdominal pain, moderate injuries, high fever
    Target: <30 minutes to physician

    Level 4 - LESS URGENT (Low Acuity)
    ──────────────────────────────────
    Stable, needs 1 resource
    Examples: Simple laceration, urinary symptoms, mild rash
    Target: <60 minutes to physician

    Level 5 - NON-URGENT (Minor)
    ────────────────────────────
    Stable, needs no resources (Rx refill, simple wound check)
    Examples: Medication refill, suture removal, minor cold symptoms
    Target: <120 minutes to physician

================================================================================
"""

from pydantic_settings import BaseSettings
from pydantic import Field, PostgresDsn
from typing import Optional, List
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Pydantic Settings automatically reads from environment variables,
    with support for .env files and type validation.
    """
    
    # ==========================================================================
    # SERVICE IDENTIFICATION
    # ==========================================================================
    service_name: str = Field(
        default="triage-acuity-agent",
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
    database_url: PostgresDsn = Field(
        default="postgresql://postgres:postgres@postgres:5432/hospital_ai",
        description="PostgreSQL connection string for triage data"
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
    # FEDERATED LEARNING CONFIGURATION
    # ==========================================================================
    fl_server_address: str = Field(
        default="fl-triage-server:8086",
        description="Flower FL server address for federated triage model training"
    )
    fl_enabled: bool = Field(
        default=True,
        description="Whether federated learning is enabled"
    )
    fl_local_epochs: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of local training epochs per FL round"
    )
    fl_min_fit_clients: int = Field(
        default=2,
        description="Minimum clients required for FL round"
    )
    hospital_id: str = Field(
        default="hospital_default",
        description="Unique hospital identifier for federated learning"
    )
    
    # ==========================================================================
    # XGBOOST MODEL CONFIGURATION
    # ==========================================================================
    xgboost_n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of boosting rounds (trees)"
    )
    xgboost_max_depth: int = Field(
        default=6,
        ge=1,
        le=15,
        description="Maximum tree depth (controls model complexity)"
    )
    xgboost_learning_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=1.0,
        description="Learning rate (eta) for gradient boosting"
    )
    xgboost_min_child_weight: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Minimum sum of instance weight needed in a child"
    )
    xgboost_subsample: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Subsample ratio of training instances"
    )
    xgboost_colsample_bytree: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Subsample ratio of columns when constructing each tree"
    )
    
    # ==========================================================================
    # RED FLAG SAFETY CONFIGURATION
    # ==========================================================================
    red_flag_override_enabled: bool = Field(
        default=True,
        description="Enable Red Flag keyword override (CRITICAL FOR PATIENT SAFETY)"
    )
    red_flag_default_acuity: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Acuity level to assign when Red Flag detected (1 or 2)"
    )
    red_flag_audit_logging: bool = Field(
        default=True,
        description="Log all Red Flag detections for clinical audit"
    )
    
    # ==========================================================================
    # ACUITY CLASSIFICATION THRESHOLDS
    # ==========================================================================
    acuity_critical_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Probability threshold for escalating to critical"
    )
    acuity_escalation_enabled: bool = Field(
        default=True,
        description="Enable automatic escalation for borderline cases"
    )
    
    # ==========================================================================
    # API CONFIGURATION
    # ==========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8005,
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
    # MLFLOW CONFIGURATION
    # ==========================================================================
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000",
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="triage_acuity",
        description="MLflow experiment name for triage models"
    )
    mlflow_model_registry_name: str = Field(
        default="triage-acuity-xgboost",
        description="Model name in MLflow Model Registry"
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
    """
    return Settings()


# ==========================================================================
# CONVENIENCE EXPORTS
# ==========================================================================
settings = get_settings()


# ==========================================================================
# ACUITY LEVEL CONSTANTS
# ==========================================================================
class AcuityLevel:
    """
    Emergency Severity Index (ESI) constants.
    
    These are the standard 5-level ESI acuity classifications used
    in Emergency Departments worldwide.
    """
    CRITICAL = 1        # Resuscitation - immediate life-saving intervention
    EMERGENT = 2        # Emergent - high risk, don't delay
    URGENT = 3          # Urgent - stable but needs multiple resources  
    LESS_URGENT = 4     # Less Urgent - stable, single resource
    NON_URGENT = 5      # Non-Urgent - could be seen in clinic
    
    # Human-readable labels
    LABELS = {
        1: "Critical (Resuscitation)",
        2: "Emergent (High Risk)",
        3: "Urgent (Moderate)",
        4: "Less Urgent (Low Acuity)",
        5: "Non-Urgent (Minor)",
    }
    
    # Target response times in minutes
    TARGET_TIMES = {
        1: 0,     # Immediate
        2: 10,    # Within 10 minutes
        3: 30,    # Within 30 minutes
        4: 60,    # Within 60 minutes
        5: 120,   # Within 120 minutes
    }
    
    @classmethod
    def get_label(cls, level: int) -> str:
        """Get human-readable label for acuity level."""
        return cls.LABELS.get(level, f"Unknown ({level})")
    
    @classmethod
    def get_target_time(cls, level: int) -> int:
        """Get target response time in minutes for acuity level."""
        return cls.TARGET_TIMES.get(level, 120)
