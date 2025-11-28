"""
ER/OR Scheduling Agent

A Hybrid Agent that combines Machine Learning (surgery duration prediction)
with Constraint Programming (OR-Tools optimization) to manage:
- Emergency Room triage queue ordering
- Operating Room block scheduling

This agent demonstrates the integration of ML predictions directly into
an optimization loop for real-time healthcare resource allocation.
"""

__version__ = "1.0.0"
__agent_type__ = "hybrid_ml_optimization"
