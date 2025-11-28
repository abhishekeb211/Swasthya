"""
Discharge Planning Agent
========================

A hybrid ML + clinical rules system for evaluating inpatient discharge readiness.

This agent combines:
1. XGBoost ML model for readiness scoring (0.0-1.0)
2. Clinical Rule Engine for safety guardrails

Key Design Principle:
    Explicit clinical rules can ALWAYS veto ML recommendations.
    This ensures patient safety and regulatory compliance.

Components:
-----------
- config: Environment configuration and clinical thresholds
- model: DischargeAnalyzer, DischargeReadinessModel, ClinicalRuleEngine
- api: FastAPI REST endpoints

Endpoints:
----------
- POST /analyze: Batch analysis of all inpatients
- POST /analyze-single: Detailed breakdown for one patient
- GET /health: Service health check
- GET /rules: List all clinical rules
- GET /model/info: ML model information

Usage Example:
--------------
```python
from discharge_planning.model import DischargeAnalyzer

analyzer = DischargeAnalyzer()
patient_data = {
    "patient_id": "P12345",
    "temperature_celsius": 37.2,
    "heart_rate": 78,
    # ... more clinical data
}
assessment = analyzer.analyze_patient(patient_data)
print(f"Recommendation: {assessment.recommendation}")
print(f"ML Score: {assessment.ml_readiness_score}")
for veto in assessment.rule_vetoes:
    print(f"VETO: {veto.reason}")
```

Port: 8004

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Hospital AI Team"
