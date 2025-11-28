"""
Federated Learning Server Package

This package contains the FL server implementations:
- demand_server: Server for demand forecasting (Prophet) models
- triage_server: Server for triage prediction (XGBoost) models
- strategy: Custom aggregation strategies (BestModelStrategy)
"""

from .strategy import BestModelStrategy, create_demand_strategy, create_triage_strategy

__all__ = [
    "BestModelStrategy",
    "create_demand_strategy",
    "create_triage_strategy",
]
