"""
Demand Forecast Agent

A microservice for predicting hospital patient volume using Facebook Prophet.

This agent provides:
- Time series forecasting with multiple seasonalities
- MLflow integration for experiment tracking
- Federated learning support for privacy-preserving training
- REST API for predictions and training triggers

Components:
-----------
- config: Environment-based configuration
- model: DemandForecaster class wrapping Prophet
- mlflow_tracking: MLflow integration utilities
- train: Local training pipeline
- fl_client: Federated learning client
- api: FastAPI application

Usage:
------
    # As API server
    python -m uvicorn api:app --host 0.0.0.0 --port 8001
    
    # As FL client
    python fl_client.py --server fl-server:8087
    
    # Training only
    python train.py --synthetic

Author: Hospital AI Platform Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Hospital AI Platform Team"

from .config import settings
from .model import DemandForecaster

__all__ = [
    "settings",
    "DemandForecaster",
    "__version__",
]
