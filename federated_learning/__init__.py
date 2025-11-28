"""
Federated Learning System for Hospital AI Platform

================================================================================
FEDERATED LEARNING: COLLABORATIVE AI WITHOUT DATA SHARING
================================================================================

This package implements a federated learning system that enables multiple
hospitals to collaboratively train AI models WITHOUT sharing patient data.

CORE CONCEPT: DATA SOVEREIGNTY
──────────────────────────────────────────────────────────────────────────────

In traditional machine learning, data is centralized:

    Hospital A ──┐
    Hospital B ──┼──► Central Server (trains model, has ALL data)
    Hospital C ──┘

    PROBLEMS:
    ✗ HIPAA/GDPR violations - patient data leaves hospital
    ✗ Single point of failure - breach exposes all data
    ✗ Trust issues - hospitals may be competitors
    ✗ Bandwidth - terabytes of medical records transferred

With federated learning, CODE travels to DATA (not vice versa):

    ┌──────────────────────────────────────────────────────────────────────────┐
    │                           FL SERVER                                       │
    │                                                                          │
    │   • Coordinates training rounds                                           │
    │   • Aggregates model PARAMETERS (not data)                               │
    │   • Never sees any patient information                                   │
    │   • Uses BestModelStrategy for smart aggregation                         │
    └──────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │   Hospital A     │  │   Hospital B     │  │   Hospital C     │
    │                  │  │                  │  │                  │
    │ [Patient Data]   │  │ [Patient Data]   │  │ [Patient Data]   │
    │      ↓           │  │      ↓           │  │      ↓           │
    │ [Local Model]    │  │ [Local Model]    │  │ [Local Model]    │
    │      ↓           │  │      ↓           │  │      ↓           │
    │ Parameters only→ │  │ Parameters only→ │  │ Parameters only→ │
    └──────────────────┘  └──────────────────┘  └──────────────────┘

    BENEFITS:
    ✓ Patient data NEVER leaves the hospital
    ✓ HIPAA/GDPR compliant by design
    ✓ Hospitals benefit from collective learning
    ✓ Works even between competing institutions

PACKAGE STRUCTURE
──────────────────────────────────────────────────────────────────────────────

federated_learning/
├── server/
│   ├── strategy.py          # BestModelStrategy - smart aggregation
│   ├── demand_server.py     # FL server for demand forecasting (port 8087)
│   └── triage_server.py     # FL server for triage prediction (port 8086)
│
├── client/
│   ├── serde.py            # Model serialization utilities
│   ├── demand_client.py    # Prophet-based demand forecasting client
│   └── triage_client.py    # XGBoost-based triage prediction client
│
├── Dockerfile.server       # Server container (lightweight)
├── Dockerfile.client       # Client container (includes ML libs)
├── requirements-server.txt # Server dependencies
└── requirements-client.txt # Client dependencies (Prophet, XGBoost)

QUICK START
──────────────────────────────────────────────────────────────────────────────

1. Start the FL Server:

    docker-compose up fl-demand-server

    Or manually:
    
    python -m federated_learning.server.demand_server --rounds 5 --min-clients 2

2. Start FL Clients (one per hospital):

    # Hospital A
    docker run -e HOSPITAL_ID=hospital_a \\
               -v /path/to/data.csv:/data/patient_volumes.csv:ro \\
               fl-client

    # Hospital B  
    docker run -e HOSPITAL_ID=hospital_b \\
               -v /path/to/data.csv:/data/patient_volumes.csv:ro \\
               fl-client

3. Monitor Training:

    The server logs aggregation progress and metrics for each round.
    Global model is saved after training completes.

MODEL TYPES
──────────────────────────────────────────────────────────────────────────────

DEMAND FORECASTING (Prophet):
    • Predicts patient volume (hourly)
    • Time-series model with seasonality
    • Server port: 8087
    • Metric: MAE (Mean Absolute Error)

TRIAGE PREDICTION (XGBoost):
    • Predicts patient acuity (ESI 1-5)
    • Classification model
    • Server port: 8086
    • Metric: Accuracy

AGGREGATION STRATEGY
──────────────────────────────────────────────────────────────────────────────

We use BestModelStrategy which improves on standard FedAvg:

Standard FedAvg:
    weight_k = num_samples_k / total_samples

BestModelStrategy:
    weight_k = performance_score_k × reliability_k × sample_weight_k

Features:
    • Performance-weighted: Better models contribute more
    • Reliability tracking: Consistent clients get higher weight
    • Anomaly detection: Corrupted updates are filtered
    • Adaptive: Learns from training history

================================================================================
"""

__version__ = "1.0.0"
__author__ = "Hospital AI Platform Team"

# Import key components for easier access
try:
    from .server.strategy import BestModelStrategy, create_demand_strategy, create_triage_strategy
    from .client.serde import (
        ModelSerializer,
        ProphetSerializer,
        XGBoostSerializer,
    )
except ImportError:
    # Allow import even if dependencies not installed
    pass

__all__ = [
    "BestModelStrategy",
    "create_demand_strategy",
    "create_triage_strategy",
    "ModelSerializer",
    "ProphetSerializer",
    "XGBoostSerializer",
]
