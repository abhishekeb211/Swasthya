"""
Demand Forecast Agent - Configuration Module

This module centralizes all environment-based configuration for the demand
forecasting microservice. It follows the 12-factor app methodology by 
externalizing configuration through environment variables.

Architecture Note:
------------------
The agent operates in two modes:
1. STANDALONE: Uses local MLflow for experimentation
2. FEDERATED: Connects to central FL server for collaborative learning

Both modes share the same database connection but differ in model storage
and training coordination.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, PostgresDsn
from typing import Optional
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
        default="demand-forecast-agent",
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
    # PostgreSQL connection for fetching historical patient data
    database_url: PostgresDsn = Field(
        default="postgresql://postgres:postgres@postgres:5432/hospital_ai",
        description="PostgreSQL connection string for patient volume data"
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
    # MLFLOW CONFIGURATION
    # ==========================================================================
    # MLflow is used for experiment tracking and model registry
    mlflow_tracking_uri: str = Field(
        default="http://mlflow:5000",
        description="MLflow tracking server URI for experiment logging"
    )
    mlflow_experiment_name: str = Field(
        default="demand_forecast",
        description="MLflow experiment name for organizing runs"
    )
    mlflow_model_registry_name: str = Field(
        default="demand-forecast-prophet",
        description="Model name in MLflow Model Registry"
    )
    
    # ==========================================================================
    # FEDERATED LEARNING CONFIGURATION
    # ==========================================================================
    # Flower (flwr) is used for federated learning across hospital network
    fl_server_address: str = Field(
        default="fl-demand-server:8087",
        description="Flower FL server address for federated training"
    )
    fl_enabled: bool = Field(
        default=True,
        description="Whether federated learning is enabled"
    )
    fl_min_fit_clients: int = Field(
        default=2,
        description="Minimum clients required for FL round"
    )
    fl_local_epochs: int = Field(
        default=3,
        description="Number of local training epochs per FL round"
    )
    
    # ==========================================================================
    # MODEL CONFIGURATION
    # ==========================================================================
    # Prophet-specific hyperparameters
    prophet_yearly_seasonality: bool = Field(
        default=True,
        description="Enable yearly seasonality in Prophet"
    )
    prophet_weekly_seasonality: bool = Field(
        default=True,
        description="Enable weekly seasonality (critical for ER patterns)"
    )
    prophet_daily_seasonality: bool = Field(
        default=True,
        description="Enable daily seasonality for hourly predictions"
    )
    prophet_changepoint_prior_scale: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Flexibility of trend changes (higher = more flexible)"
    )
    prophet_seasonality_prior_scale: float = Field(
        default=10.0,
        ge=0.01,
        le=100.0,
        description="Strength of seasonality (higher = stronger)"
    )
    
    # Forecast configuration
    forecast_horizon_hours: int = Field(
        default=168,  # 7 days
        description="Default forecast horizon in hours"
    )
    forecast_frequency: str = Field(
        default="H",  # Hourly
        description="Forecast frequency (H=hourly, D=daily)"
    )
    
    # ==========================================================================
    # API CONFIGURATION
    # ==========================================================================
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    api_port: int = Field(
        default=8001,
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
        # Allow extra fields for forward compatibility
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
