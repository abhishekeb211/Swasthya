"""
Triage & Acuity Agent
=====================

AI-powered patient triage system for Emergency Department operations.

This agent analyzes patient symptoms and vital signs to assign an Emergency
Severity Index (ESI) acuity level from 1 (Critical/Resuscitation) to 5 (Non-urgent).

Key Features:
- XGBoost-based acuity classification
- Red Flag keyword detection for safety overrides
- Federated Learning for privacy-preserving training
- Active Learning through clinician override feedback

Port: 8005
FL Server: fl-triage-server:8086
"""

__version__ = "1.0.0"
__author__ = "Hospital AI Team"
