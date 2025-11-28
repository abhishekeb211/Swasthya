"""
Federated Learning Client Package

This package contains FL client implementations:
- demand_client: Client for demand forecasting (Prophet) models
- triage_client: Client for triage prediction (XGBoost) models
- serde: Serialization utilities for model transmission
"""

from .serde import (
    ModelSerializer,
    ProphetSerializer,
    XGBoostSerializer,
    serialize_model,
    deserialize_model,
    get_model_parameters,
    set_model_parameters,
)

__all__ = [
    "ModelSerializer",
    "ProphetSerializer",
    "XGBoostSerializer",
    "serialize_model",
    "deserialize_model",
    "get_model_parameters",
    "set_model_parameters",
]
