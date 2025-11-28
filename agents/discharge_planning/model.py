"""
Discharge Planning Agent - Hybrid ML + Rule Engine Model

This module implements the core discharge readiness prediction system using a
HYBRID APPROACH that combines machine learning with explicit clinical rules.

================================================================================
CRITICAL ARCHITECTURAL DECISION: Why "Black Box" AI is INSUFFICIENT for 
Discharge Decisions
================================================================================

Unlike demand forecasting where prediction errors result in minor inefficiencies,
discharge planning errors have DIRECT PATIENT SAFETY IMPLICATIONS:

1. LIABILITY AND LEGAL EXPOSURE:
   ─────────────────────────────
   If a patient is discharged based on an ML model's recommendation and 
   subsequently experiences adverse outcomes (readmission, death), the hospital
   faces significant legal exposure. Courts require EXPLAINABLE decision-making.
   
   "The algorithm said so" is NOT a valid legal defense.
   
   Explicit rules provide:
   - Auditable decision trails
   - Clear clinical rationale
   - Defensible discharge criteria

2. REGULATORY COMPLIANCE:
   ───────────────────────
   Healthcare regulations (CMS, Joint Commission) require documented clinical
   criteria for discharge. Black-box ML models cannot satisfy these requirements:
   
   - CMS Conditions of Participation require "discharge planning evaluation"
   - Joint Commission requires documented discharge criteria
   - State regulations often mandate specific clinical thresholds
   
3. RARE BUT CRITICAL EVENTS:
   ─────────────────────────
   ML models learn from DATA DISTRIBUTIONS. Rare but critical contraindications
   may be underrepresented in training data:
   
   - Post-surgical bleeding complications (rare but deadly)
   - Drug-drug interactions specific to discharge medications
   - Fall risk in elderly patients being sent to non-supervised homes
   
   Explicit rules GUARANTEE these are always checked, regardless of training data.

4. CLINICAL WORKFLOW TRUST:
   ─────────────────────────
   Physicians and nurses will NOT trust discharge recommendations they cannot
   understand. The hybrid approach allows:
   
   - ML model provides a "baseline readiness score" 
   - Rules provide "vetoes" with explicit explanations
   - Clinicians can see exactly WHY a patient is/isn't ready
   
5. CONCEPT DRIFT RESILIENCE:
   ─────────────────────────
   Patient populations change (aging, new diseases, pandemic shifts). ML models
   can silently degrade. Explicit rules remain valid because they encode
   PHYSIOLOGICAL CONSTANTS (e.g., fever > 38°C indicates infection).

================================================================================
HYBRID ARCHITECTURE
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         PATIENT CLINICAL DATA                           │
    │   (vitals, labs, diagnoses, procedures, medications, social factors)    │
    └─────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
           ┌───────────────────────────────────────────────────────┐
           │                                                       │
           │   ┌─────────────────┐         ┌─────────────────┐    │
           │   │    XGBoost      │         │   Rule Engine   │    │
           │   │  Readiness      │         │   (Clinical     │    │
           │   │  Score Model    │         │   Guardrails)   │    │
           │   │                 │         │                 │    │
           │   │  Learns from:   │         │  Hard-coded:    │    │
           │   │  - LOS patterns │         │  - Vital limits │    │
           │   │  - Readmissions │         │  - Lab ranges   │    │
           │   │  - Outcomes     │         │  - Pending tests│    │
           │   │  - Demographics │         │  - Safety rules │    │
           │   │                 │         │                 │    │
           │   │  Output: 0.0-1.0│         │  Output: VETO   │    │
           │   │  (probability)  │         │  or APPROVE     │    │
           │   └────────┬────────┘         └────────┬────────┘    │
           │            │                           │              │
           │            ▼                           ▼              │
           │   ┌─────────────────────────────────────────────┐    │
           │   │              DECISION LOGIC                 │    │
           │   │                                             │    │
           │   │  IF any_rule_vetoes:                        │    │
           │   │      recommendation = "NOT_READY"           │    │
           │   │      reason = rule_violation_explanation    │    │
           │   │  ELIF readiness_score >= 0.7:               │    │
           │   │      recommendation = "LIKELY_READY"        │    │
           │   │  ELIF readiness_score >= 0.4:               │    │
           │   │      recommendation = "NEEDS_REVIEW"        │    │
           │   │  ELSE:                                      │    │
           │   │      recommendation = "NOT_READY"           │    │
           │   └─────────────────────────────────────────────┘    │
           │                                                       │
           └───────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     EXPLAINABLE RECOMMENDATION                          │
    │                                                                         │
    │  {                                                                      │
    │    "patient_id": "P12345",                                              │
    │    "recommendation": "NOT_READY",                                       │
    │    "ml_readiness_score": 0.82,                                          │
    │    "rule_vetoes": [                                                     │
    │      {                                                                  │
    │        "rule": "FEVER_CHECK",                                           │
    │        "reason": "Temperature 38.7°C exceeds safe threshold (38.0°C)", │
    │        "severity": "CRITICAL"                                           │
    │      }                                                                  │
    │    ],                                                                   │
    │    "contributing_factors": [                                            │
    │      {"factor": "stable_vitals_48h", "contribution": +0.15},            │
    │      {"factor": "completed_physical_therapy", "contribution": +0.12}   │
    │    ]                                                                    │
    │  }                                                                      │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config import settings

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS AND DATA CLASSES
# =============================================================================

class DischargeRecommendation(str, Enum):
    """
    Possible discharge recommendations.
    
    These map to clinical workflow stages:
    - LIKELY_READY: Add to discharge candidate list, expedite physician review
    - NEEDS_REVIEW: Flag for case manager, may need social work consult
    - NOT_READY: Continue inpatient care, no discharge action
    - VETOED: ML said ready, but clinical rules blocked discharge
    """
    LIKELY_READY = "LIKELY_READY"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NOT_READY = "NOT_READY"
    VETOED = "VETOED"


class VetoSeverity(str, Enum):
    """
    Severity level of rule violations.
    
    Used for prioritizing which issues to address first:
    - CRITICAL: Immediate patient safety risk, must be resolved
    - WARNING: Should be addressed but not immediately dangerous
    - INFO: Informational flag, clinician discretion
    """
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class RuleVeto:
    """
    Represents a clinical rule violation that vetoes discharge.
    
    Attributes:
        rule_id: Unique identifier for the rule (for tracking/analytics)
        rule_name: Human-readable rule name
        reason: Detailed explanation of why this rule was triggered
        severity: How critical this veto is
        actual_value: The patient's actual value that triggered the rule
        threshold_value: The threshold that was violated
        recommendation: What action should be taken
    """
    rule_id: str
    rule_name: str
    reason: str
    severity: VetoSeverity
    actual_value: Optional[Union[float, str]] = None
    threshold_value: Optional[Union[float, str]] = None
    recommendation: Optional[str] = None


@dataclass
class ContributingFactor:
    """
    Represents a factor that contributed to the ML readiness score.
    
    Used for explainability - shows clinicians which features drove the prediction.
    """
    factor_name: str
    contribution: float  # Positive = increases readiness, Negative = decreases
    description: str
    raw_value: Optional[Any] = None


@dataclass
class DischargeAssessment:
    """
    Complete discharge assessment result for a patient.
    
    This is the main output of the hybrid model, containing all information
    needed for clinical decision-making.
    """
    patient_id: str
    assessment_timestamp: datetime
    
    # Final recommendation (after rules)
    recommendation: DischargeRecommendation
    
    # ML model output
    ml_readiness_score: float  # 0.0 - 1.0
    ml_score_interpretation: str  # Human-readable interpretation
    
    # Rule engine output
    rule_vetoes: List[RuleVeto] = field(default_factory=list)
    rules_passed: int = 0
    rules_failed: int = 0
    
    # Explainability
    contributing_factors: List[ContributingFactor] = field(default_factory=list)
    
    # Additional context
    length_of_stay_hours: Optional[float] = None
    admission_type: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    
    # Metadata for tracking
    model_version: str = "1.0.0"
    rule_engine_version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "patient_id": self.patient_id,
            "assessment_timestamp": self.assessment_timestamp.isoformat(),
            "recommendation": self.recommendation.value,
            "ml_readiness_score": round(self.ml_readiness_score, 3),
            "ml_score_interpretation": self.ml_score_interpretation,
            "rule_vetoes": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "reason": v.reason,
                    "severity": v.severity.value,
                    "actual_value": v.actual_value,
                    "threshold_value": v.threshold_value,
                    "recommendation": v.recommendation,
                }
                for v in self.rule_vetoes
            ],
            "rules_passed": self.rules_passed,
            "rules_failed": self.rules_failed,
            "contributing_factors": [
                {
                    "factor_name": f.factor_name,
                    "contribution": round(f.contribution, 3),
                    "description": f.description,
                    "raw_value": f.raw_value,
                }
                for f in self.contributing_factors
            ],
            "length_of_stay_hours": self.length_of_stay_hours,
            "admission_type": self.admission_type,
            "primary_diagnosis": self.primary_diagnosis,
            "model_version": self.model_version,
            "rule_engine_version": self.rule_engine_version,
        }


# =============================================================================
# CLINICAL RULE ENGINE
# =============================================================================

class ClinicalRuleEngine:
    """
    Rule-based clinical guardrails for discharge decisions.
    
    This class implements explicit, auditable rules that can VETO discharge
    recommendations regardless of what the ML model predicts. This is critical
    for patient safety and regulatory compliance.
    
    Rules are organized into categories:
    1. VITAL SIGN RULES: Temperature, heart rate, blood pressure, oxygen
    2. LABORATORY RULES: WBC, hemoglobin, electrolytes, kidney function
    3. OPERATIONAL RULES: Pending tests, minimum LOS, follow-up requirements
    4. SAFETY RULES: Fall risk, medication reconciliation, social factors
    
    Each rule:
    - Has a unique ID for tracking
    - Returns a RuleVeto object if violated
    - Includes human-readable explanation
    - Specifies recommended action
    
    Example:
        >>> engine = ClinicalRuleEngine()
        >>> patient_data = {"temperature_celsius": 38.7, ...}
        >>> vetoes = engine.evaluate(patient_data)
        >>> for veto in vetoes:
        ...     print(f"{veto.rule_name}: {veto.reason}")
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        """Initialize the rule engine with configurable thresholds."""
        self.settings = settings
        logger.info(f"ClinicalRuleEngine v{self.VERSION} initialized")
    
    def evaluate(self, patient_data: Dict[str, Any]) -> List[RuleVeto]:
        """
        Evaluate all clinical rules against patient data.
        
        Args:
            patient_data: Dictionary containing patient clinical information
                Expected keys:
                - temperature_celsius: float
                - heart_rate: int
                - systolic_bp: int
                - diastolic_bp: int
                - spo2: float
                - wbc_count: float
                - hemoglobin: float
                - creatinine: float
                - potassium: float
                - pending_labs: bool
                - pending_imaging: bool
                - fever_last_24h: bool
                - admission_datetime: datetime
                - admission_type: str
                - fall_risk_score: int
                - on_iv_medications: bool
                - has_foley_catheter: bool
                - has_wound_vac: bool
                - needs_home_oxygen: bool
                - social_work_cleared: bool
                - pharmacy_reconciliation_complete: bool
        
        Returns:
            List of RuleVeto objects for any violated rules
        """
        vetoes = []
        
        # Run all rule categories
        vetoes.extend(self._check_vital_sign_rules(patient_data))
        vetoes.extend(self._check_laboratory_rules(patient_data))
        vetoes.extend(self._check_operational_rules(patient_data))
        vetoes.extend(self._check_safety_rules(patient_data))
        
        if vetoes:
            logger.info(
                f"Patient {patient_data.get('patient_id', 'unknown')} has "
                f"{len(vetoes)} rule violations"
            )
        
        return vetoes
    
    def _check_vital_sign_rules(self, data: Dict[str, Any]) -> List[RuleVeto]:
        """Check vital sign-related discharge rules."""
        vetoes = []
        
        # Rule: FEVER_CHECK
        # Rationale: Fever indicates active infection or inflammatory process
        temp = data.get("temperature_celsius")
        if temp is not None and temp >= self.settings.fever_threshold_celsius:
            vetoes.append(RuleVeto(
                rule_id="VS001",
                rule_name="FEVER_CHECK",
                reason=f"Temperature {temp:.1f}°C exceeds safe discharge threshold "
                       f"({self.settings.fever_threshold_celsius}°C)",
                severity=VetoSeverity.CRITICAL,
                actual_value=temp,
                threshold_value=self.settings.fever_threshold_celsius,
                recommendation="Investigate source of fever before discharge. "
                              "Consider blood cultures if not already obtained."
            ))
        
        # Rule: FEVER_LAST_24H
        # Rationale: Recent fever may indicate unresolved infection
        if data.get("fever_last_24h"):
            vetoes.append(RuleVeto(
                rule_id="VS002",
                rule_name="RECENT_FEVER",
                reason="Patient had fever within last 24 hours",
                severity=VetoSeverity.WARNING,
                actual_value="Yes",
                threshold_value="No fever in 24h",
                recommendation="Monitor temperature for 24h afebrile period "
                              "before discharge."
            ))
        
        # Rule: TACHYCARDIA_CHECK
        # Rationale: Tachycardia may indicate pain, infection, dehydration, or cardiac issues
        hr = data.get("heart_rate")
        if hr is not None and hr > self.settings.tachycardia_threshold:
            vetoes.append(RuleVeto(
                rule_id="VS003",
                rule_name="TACHYCARDIA_CHECK",
                reason=f"Heart rate {hr} bpm exceeds safe threshold "
                       f"({self.settings.tachycardia_threshold} bpm)",
                severity=VetoSeverity.WARNING,
                actual_value=hr,
                threshold_value=self.settings.tachycardia_threshold,
                recommendation="Evaluate cause of tachycardia. Consider "
                              "pain assessment, hydration status, and EKG."
            ))
        
        # Rule: BRADYCARDIA_CHECK
        # Rationale: Bradycardia may indicate heart block or medication effect
        if hr is not None and hr < self.settings.bradycardia_threshold:
            vetoes.append(RuleVeto(
                rule_id="VS004",
                rule_name="BRADYCARDIA_CHECK",
                reason=f"Heart rate {hr} bpm is below safe threshold "
                       f"({self.settings.bradycardia_threshold} bpm)",
                severity=VetoSeverity.WARNING,
                actual_value=hr,
                threshold_value=self.settings.bradycardia_threshold,
                recommendation="Evaluate bradycardia. Review medications "
                              "(beta-blockers, calcium channel blockers). "
                              "Consider cardiology consult."
            ))
        
        # Rule: HYPOXIA_CHECK
        # Rationale: Low oxygen saturation indicates respiratory compromise
        spo2 = data.get("spo2")
        if spo2 is not None and spo2 < self.settings.hypoxia_threshold:
            vetoes.append(RuleVeto(
                rule_id="VS005",
                rule_name="HYPOXIA_CHECK",
                reason=f"Oxygen saturation {spo2}% is below safe threshold "
                       f"({self.settings.hypoxia_threshold}%)",
                severity=VetoSeverity.CRITICAL,
                actual_value=spo2,
                threshold_value=self.settings.hypoxia_threshold,
                recommendation="Patient requires supplemental oxygen. If home "
                              "oxygen needed, ensure equipment and education "
                              "arranged before discharge."
            ))
        
        # Rule: HYPOTENSION_CHECK
        # Rationale: Low blood pressure may indicate sepsis, bleeding, or cardiac issue
        sbp = data.get("systolic_bp")
        if sbp is not None and sbp < self.settings.hypotension_systolic_threshold:
            vetoes.append(RuleVeto(
                rule_id="VS006",
                rule_name="HYPOTENSION_CHECK",
                reason=f"Systolic blood pressure {sbp} mmHg is below safe threshold "
                       f"({self.settings.hypotension_systolic_threshold} mmHg)",
                severity=VetoSeverity.CRITICAL,
                actual_value=sbp,
                threshold_value=self.settings.hypotension_systolic_threshold,
                recommendation="Evaluate hypotension. Consider fluid resuscitation, "
                              "sepsis workup, or cardiac evaluation."
            ))
        
        # Rule: HYPERTENSIVE_CRISIS_CHECK
        # Rationale: Severely elevated BP requires immediate management
        if sbp is not None and sbp > self.settings.hypertension_systolic_threshold:
            vetoes.append(RuleVeto(
                rule_id="VS007",
                rule_name="HYPERTENSIVE_CRISIS_CHECK",
                reason=f"Systolic blood pressure {sbp} mmHg exceeds crisis threshold "
                       f"({self.settings.hypertension_systolic_threshold} mmHg)",
                severity=VetoSeverity.CRITICAL,
                actual_value=sbp,
                threshold_value=self.settings.hypertension_systolic_threshold,
                recommendation="Manage hypertensive urgency/emergency before discharge. "
                              "Optimize oral antihypertensives."
            ))
        
        return vetoes
    
    def _check_laboratory_rules(self, data: Dict[str, Any]) -> List[RuleVeto]:
        """Check laboratory value-related discharge rules."""
        vetoes = []
        
        # Rule: WBC_HIGH_CHECK
        # Rationale: Elevated WBC indicates infection or inflammation
        wbc = data.get("wbc_count")
        if wbc is not None and wbc > self.settings.wbc_high_threshold:
            vetoes.append(RuleVeto(
                rule_id="LAB001",
                rule_name="HIGH_WBC_CHECK",
                reason=f"White blood cell count {wbc:.1f} x10^9/L exceeds threshold "
                       f"({self.settings.wbc_high_threshold} x10^9/L)",
                severity=VetoSeverity.WARNING,
                actual_value=wbc,
                threshold_value=self.settings.wbc_high_threshold,
                recommendation="Investigate source of leukocytosis. Consider "
                              "infectious workup if not already completed."
            ))
        
        # Rule: WBC_LOW_CHECK
        # Rationale: Low WBC (leukopenia) indicates immune compromise
        if wbc is not None and wbc < self.settings.wbc_low_threshold:
            vetoes.append(RuleVeto(
                rule_id="LAB002",
                rule_name="LOW_WBC_CHECK",
                reason=f"White blood cell count {wbc:.1f} x10^9/L is below threshold "
                       f"({self.settings.wbc_low_threshold} x10^9/L)",
                severity=VetoSeverity.WARNING,
                actual_value=wbc,
                threshold_value=self.settings.wbc_low_threshold,
                recommendation="Patient is immunocompromised. Ensure appropriate "
                              "precautions and follow-up arranged."
            ))
        
        # Rule: HEMOGLOBIN_LOW_CHECK
        # Rationale: Severe anemia may require transfusion
        hgb = data.get("hemoglobin")
        if hgb is not None and hgb < self.settings.hemoglobin_low_threshold:
            vetoes.append(RuleVeto(
                rule_id="LAB003",
                rule_name="SEVERE_ANEMIA_CHECK",
                reason=f"Hemoglobin {hgb:.1f} g/dL is critically low "
                       f"(threshold: {self.settings.hemoglobin_low_threshold} g/dL)",
                severity=VetoSeverity.CRITICAL,
                actual_value=hgb,
                threshold_value=self.settings.hemoglobin_low_threshold,
                recommendation="Consider blood transfusion before discharge. "
                              "Ensure outpatient hematology follow-up."
            ))
        
        # Rule: CREATININE_HIGH_CHECK
        # Rationale: Elevated creatinine indicates acute kidney injury
        creat = data.get("creatinine")
        if creat is not None and creat > self.settings.creatinine_high_threshold:
            vetoes.append(RuleVeto(
                rule_id="LAB004",
                rule_name="KIDNEY_FUNCTION_CHECK",
                reason=f"Creatinine {creat:.2f} mg/dL indicates kidney impairment "
                       f"(threshold: {self.settings.creatinine_high_threshold} mg/dL)",
                severity=VetoSeverity.WARNING,
                actual_value=creat,
                threshold_value=self.settings.creatinine_high_threshold,
                recommendation="Evaluate cause of elevated creatinine. Review "
                              "nephrotoxic medications. Consider nephrology consult."
            ))
        
        # Rule: POTASSIUM_LOW_CHECK
        # Rationale: Hypokalemia can cause arrhythmias
        k = data.get("potassium")
        if k is not None and k < self.settings.potassium_low_threshold:
            vetoes.append(RuleVeto(
                rule_id="LAB005",
                rule_name="HYPOKALEMIA_CHECK",
                reason=f"Potassium {k:.1f} mEq/L is dangerously low "
                       f"(threshold: {self.settings.potassium_low_threshold} mEq/L)",
                severity=VetoSeverity.CRITICAL,
                actual_value=k,
                threshold_value=self.settings.potassium_low_threshold,
                recommendation="Replete potassium before discharge. Check EKG "
                              "and recheck level after repletion."
            ))
        
        # Rule: POTASSIUM_HIGH_CHECK
        # Rationale: Hyperkalemia can cause fatal arrhythmias
        if k is not None and k > self.settings.potassium_high_threshold:
            vetoes.append(RuleVeto(
                rule_id="LAB006",
                rule_name="HYPERKALEMIA_CHECK",
                reason=f"Potassium {k:.1f} mEq/L is dangerously high "
                       f"(threshold: {self.settings.potassium_high_threshold} mEq/L)",
                severity=VetoSeverity.CRITICAL,
                actual_value=k,
                threshold_value=self.settings.potassium_high_threshold,
                recommendation="Treat hyperkalemia urgently. Check EKG for "
                              "cardiac effects. Consider nephrology consult."
            ))
        
        return vetoes
    
    def _check_operational_rules(self, data: Dict[str, Any]) -> List[RuleVeto]:
        """Check operational/workflow-related discharge rules."""
        vetoes = []
        
        # Rule: PENDING_LABS_CHECK
        # Rationale: Results may reveal conditions requiring continued hospitalization
        if data.get("pending_labs"):
            vetoes.append(RuleVeto(
                rule_id="OP001",
                rule_name="PENDING_LABS_CHECK",
                reason="Patient has pending laboratory results",
                severity=VetoSeverity.WARNING,
                actual_value="Pending",
                threshold_value="All labs resulted",
                recommendation="Wait for lab results before discharge or ensure "
                              "reliable outpatient follow-up mechanism."
            ))
        
        # Rule: PENDING_IMAGING_CHECK
        # Rationale: Imaging may reveal surgical emergency or new diagnosis
        if data.get("pending_imaging"):
            vetoes.append(RuleVeto(
                rule_id="OP002",
                rule_name="PENDING_IMAGING_CHECK",
                reason="Patient has pending imaging studies",
                severity=VetoSeverity.WARNING,
                actual_value="Pending",
                threshold_value="All imaging completed",
                recommendation="Review imaging results before discharge. "
                              "Critical findings may change management."
            ))
        
        # Rule: MINIMUM_LOS_CHECK
        # Rationale: Certain admission types require minimum observation periods
        admission_dt = data.get("admission_datetime")
        admission_type = data.get("admission_type", "").lower()
        
        if admission_dt:
            los_hours = (datetime.utcnow() - admission_dt).total_seconds() / 3600
            
            min_los = None
            if "emergency" in admission_type or "er" in admission_type:
                min_los = self.settings.min_los_hours_emergency
            elif "surg" in admission_type:
                min_los = self.settings.min_los_hours_surgical
            elif "observation" in admission_type or "obs" in admission_type:
                min_los = self.settings.min_los_hours_observation
            
            if min_los and los_hours < min_los:
                vetoes.append(RuleVeto(
                    rule_id="OP003",
                    rule_name="MINIMUM_LOS_CHECK",
                    reason=f"Length of stay {los_hours:.1f} hours is below minimum "
                           f"for {admission_type} admission ({min_los} hours required)",
                    severity=VetoSeverity.WARNING,
                    actual_value=f"{los_hours:.1f} hours",
                    threshold_value=f"{min_los} hours",
                    recommendation="Patient may be discharged after minimum "
                                  f"observation period ({min_los} hours)."
                ))
        
        return vetoes
    
    def _check_safety_rules(self, data: Dict[str, Any]) -> List[RuleVeto]:
        """Check patient safety-related discharge rules."""
        vetoes = []
        
        # Rule: IV_MEDICATION_CHECK
        # Rationale: IV medications generally require hospital administration
        if data.get("on_iv_medications"):
            vetoes.append(RuleVeto(
                rule_id="SF001",
                rule_name="IV_MEDICATION_CHECK",
                reason="Patient is receiving IV medications",
                severity=VetoSeverity.WARNING,
                actual_value="Yes",
                threshold_value="No IV medications",
                recommendation="Convert to oral medications before discharge, "
                              "or arrange home infusion therapy if indicated."
            ))
        
        # Rule: FOLEY_CATHETER_CHECK
        # Rationale: Foley catheter requires education and follow-up
        if data.get("has_foley_catheter"):
            vetoes.append(RuleVeto(
                rule_id="SF002",
                rule_name="FOLEY_CATHETER_CHECK",
                reason="Patient has indwelling urinary catheter",
                severity=VetoSeverity.INFO,
                actual_value="Yes",
                threshold_value="No catheter",
                recommendation="Ensure catheter care education completed and "
                              "urology follow-up scheduled."
            ))
        
        # Rule: FALL_RISK_CHECK
        # Rationale: High fall risk patients need safe disposition planning
        fall_risk = data.get("fall_risk_score", 0)
        if fall_risk >= 3:  # Typically 0-5 scale where 3+ is high risk
            vetoes.append(RuleVeto(
                rule_id="SF003",
                rule_name="HIGH_FALL_RISK_CHECK",
                reason=f"Patient has high fall risk score ({fall_risk}/5)",
                severity=VetoSeverity.WARNING,
                actual_value=fall_risk,
                threshold_value=3,
                recommendation="Ensure safe discharge disposition. Consider "
                              "PT evaluation, home safety assessment, and "
                              "possible SNF if home unsafe."
            ))
        
        # Rule: PHARMACY_RECONCILIATION_CHECK
        # Rationale: Medication errors at discharge are common and dangerous
        if not data.get("pharmacy_reconciliation_complete"):
            vetoes.append(RuleVeto(
                rule_id="SF004",
                rule_name="PHARMACY_RECONCILIATION_CHECK",
                reason="Pharmacy medication reconciliation not completed",
                severity=VetoSeverity.WARNING,
                actual_value="Incomplete",
                threshold_value="Complete",
                recommendation="Complete pharmacy reconciliation before discharge "
                              "to prevent medication errors."
            ))
        
        # Rule: SOCIAL_WORK_CLEARANCE_CHECK
        # Rationale: Patients with complex social situations need safe discharge plans
        if data.get("needs_social_work") and not data.get("social_work_cleared"):
            vetoes.append(RuleVeto(
                rule_id="SF005",
                rule_name="SOCIAL_WORK_CLEARANCE_CHECK",
                reason="Social work clearance pending for this patient",
                severity=VetoSeverity.WARNING,
                actual_value="Pending",
                threshold_value="Cleared",
                recommendation="Await social work assessment to ensure safe "
                              "discharge disposition and follow-up."
            ))
        
        return vetoes


# =============================================================================
# XGBOOST READINESS MODEL
# =============================================================================

class DischargeReadinessModel:
    """
    XGBoost model for predicting discharge readiness score.
    
    This model learns patterns from historical discharge data to predict
    how "ready" a patient is for discharge. The score is used as one input
    to the final decision (along with rule engine vetoes).
    
    The model is trained on features including:
    - Length of stay and trends
    - Vital sign stability
    - Laboratory values and trends
    - Procedure history
    - Demographics and comorbidities
    - Historical readmission patterns
    
    IMPORTANT: This score alone does NOT determine discharge eligibility.
    Clinical rules can always veto a high score.
    
    Attributes:
        model: Trained XGBoost classifier
        scaler: Feature scaler for normalization
        feature_names: List of expected feature names
        is_fitted: Whether model has been trained
    """
    
    # Feature definitions with descriptions (for explainability)
    FEATURE_DEFINITIONS = {
        "los_hours": "Length of stay in hours",
        "age": "Patient age in years",
        "temperature_celsius": "Most recent temperature (°C)",
        "heart_rate": "Most recent heart rate (bpm)",
        "systolic_bp": "Most recent systolic blood pressure (mmHg)",
        "diastolic_bp": "Most recent diastolic blood pressure (mmHg)",
        "spo2": "Most recent oxygen saturation (%)",
        "respiratory_rate": "Most recent respiratory rate (breaths/min)",
        "wbc_count": "Most recent WBC count (x10^9/L)",
        "hemoglobin": "Most recent hemoglobin (g/dL)",
        "creatinine": "Most recent creatinine (mg/dL)",
        "potassium": "Most recent potassium (mEq/L)",
        "sodium": "Most recent sodium (mEq/L)",
        "glucose": "Most recent glucose (mg/dL)",
        "vital_stability_score": "Stability of vitals over last 24h (0-100)",
        "mobility_score": "Physical therapy mobility assessment (0-10)",
        "pain_score": "Current pain level (0-10)",
        "num_active_diagnoses": "Number of active diagnoses",
        "num_procedures_48h": "Procedures in last 48 hours",
        "charlson_comorbidity_index": "Comorbidity burden score",
        "prior_readmissions_30d": "Readmissions in prior 30 days",
        "has_caregiver_at_home": "Whether patient has home caregiver",
        "ambulatory_status": "Can patient ambulate independently (0/1)",
        "tolerating_oral_intake": "Tolerating oral diet (0/1)",
    }
    
    def __init__(self):
        """Initialize the readiness model with configured hyperparameters."""
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names = list(self.FEATURE_DEFINITIONS.keys())
        self.is_fitted = False
        self.model_metadata = {}
        
        # Initialize XGBoost with configured parameters
        self.model = xgb.XGBClassifier(
            n_estimators=settings.xgb_n_estimators,
            max_depth=settings.xgb_max_depth,
            learning_rate=settings.xgb_learning_rate,
            min_child_weight=settings.xgb_min_child_weight,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=42,
        )
        
        logger.info("DischargeReadinessModel initialized")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "DischargeReadinessModel":
        """
        Train the readiness model.
        
        Args:
            X: Feature DataFrame with columns matching FEATURE_DEFINITIONS
            y: Binary target (1 = successful discharge, 0 = readmission/complication)
            validation_data: Optional (X_val, y_val) for early stopping
        
        Returns:
            self for method chaining
        """
        logger.info(f"Training readiness model on {len(X)} samples")
        
        # Align features
        X_aligned = self._align_features(X)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_aligned)
        
        # Prepare eval set if provided
        eval_set = None
        if validation_data:
            X_val_aligned = self._align_features(validation_data[0])
            X_val_scaled = self.scaler.transform(X_val_aligned)
            eval_set = [(X_val_scaled, validation_data[1])]
        
        # Train model
        self.model.fit(
            X_scaled,
            y,
            eval_set=eval_set,
            verbose=False,
        )
        
        self.is_fitted = True
        self.model_metadata = {
            "trained_at": datetime.utcnow().isoformat(),
            "n_samples": len(X),
            "n_features": len(self.feature_names),
            "positive_rate": float(y.mean()),
        }
        
        logger.info(f"Model training complete. Feature importances calculated.")
        
        return self
    
    def predict_readiness(
        self,
        patient_data: Dict[str, Any],
    ) -> Tuple[float, List[ContributingFactor]]:
        """
        Predict discharge readiness score for a patient.
        
        Args:
            patient_data: Dictionary with patient features
        
        Returns:
            Tuple of (readiness_score, contributing_factors)
            - readiness_score: Float 0.0-1.0 (probability of successful discharge)
            - contributing_factors: List of factors that influenced the score
        """
        if not self.is_fitted:
            # Return synthetic score if model not trained
            logger.warning("Model not fitted, returning synthetic score")
            return self._synthetic_prediction(patient_data)
        
        # Prepare features
        X = pd.DataFrame([patient_data])
        X_aligned = self._align_features(X)
        X_scaled = self.scaler.transform(X_aligned)
        
        # Get probability
        readiness_score = float(self.model.predict_proba(X_scaled)[0, 1])
        
        # Calculate feature contributions using SHAP-like approach
        contributing_factors = self._calculate_contributions(
            patient_data, X_scaled[0], readiness_score
        )
        
        return readiness_score, contributing_factors
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has all expected features in correct order."""
        result = pd.DataFrame(index=X.index)
        
        for feature in self.feature_names:
            if feature in X.columns:
                result[feature] = X[feature]
            else:
                result[feature] = 0  # Default value for missing features
        
        return result
    
    def _calculate_contributions(
        self,
        raw_data: Dict[str, Any],
        scaled_features: np.ndarray,
        score: float,
    ) -> List[ContributingFactor]:
        """
        Calculate which features contributed most to the prediction.
        
        Uses feature importances as a proxy for SHAP values.
        In production, consider using actual SHAP for better accuracy.
        """
        contributions = []
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Calculate contribution direction based on feature value relative to mean
        for i, feature_name in enumerate(self.feature_names):
            importance = importances[i]
            if importance < 0.01:  # Skip very low importance features
                continue
            
            raw_value = raw_data.get(feature_name)
            scaled_value = scaled_features[i]
            
            # Positive scaled value suggests above average, negative below
            direction = 1 if scaled_value > 0 else -1
            contribution = importance * direction * 0.5  # Scale to reasonable range
            
            description = self._get_contribution_description(
                feature_name, raw_value, contribution > 0
            )
            
            contributions.append(ContributingFactor(
                factor_name=feature_name,
                contribution=contribution,
                description=description,
                raw_value=raw_value,
            ))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x.contribution), reverse=True)
        
        # Return top 5
        return contributions[:5]
    
    def _get_contribution_description(
        self,
        feature_name: str,
        raw_value: Any,
        is_positive: bool,
    ) -> str:
        """Generate human-readable description of feature contribution."""
        base_description = self.FEATURE_DEFINITIONS.get(
            feature_name, feature_name
        )
        
        direction = "increases" if is_positive else "decreases"
        
        if raw_value is not None:
            return f"{base_description} ({raw_value}) {direction} readiness"
        else:
            return f"{base_description} {direction} readiness"
    
    def _synthetic_prediction(
        self,
        patient_data: Dict[str, Any],
    ) -> Tuple[float, List[ContributingFactor]]:
        """
        Generate a rule-based synthetic prediction when model isn't trained.
        
        This provides reasonable defaults for testing and initial deployment.
        """
        score = 0.5  # Start neutral
        contributions = []
        
        # Vital sign stability
        temp = patient_data.get("temperature_celsius", 37.0)
        if 36.5 <= temp <= 37.5:
            score += 0.1
            contributions.append(ContributingFactor(
                "temperature_celsius", 0.1,
                f"Normal temperature ({temp}°C) increases readiness",
                temp
            ))
        else:
            score -= 0.1
            contributions.append(ContributingFactor(
                "temperature_celsius", -0.1,
                f"Abnormal temperature ({temp}°C) decreases readiness",
                temp
            ))
        
        # Length of stay consideration
        los = patient_data.get("los_hours", 48)
        if los >= 24:
            score += 0.1
            contributions.append(ContributingFactor(
                "los_hours", 0.1,
                f"Adequate observation period ({los}h) increases readiness",
                los
            ))
        
        # Mobility
        mobility = patient_data.get("mobility_score", 5)
        if mobility >= 7:
            score += 0.15
            contributions.append(ContributingFactor(
                "mobility_score", 0.15,
                f"Good mobility ({mobility}/10) increases readiness",
                mobility
            ))
        elif mobility <= 3:
            score -= 0.15
            contributions.append(ContributingFactor(
                "mobility_score", -0.15,
                f"Poor mobility ({mobility}/10) decreases readiness",
                mobility
            ))
        
        # Oral intake
        if patient_data.get("tolerating_oral_intake", True):
            score += 0.1
            contributions.append(ContributingFactor(
                "tolerating_oral_intake", 0.1,
                "Tolerating oral intake increases readiness",
                True
            ))
        
        # Pain control
        pain = patient_data.get("pain_score", 3)
        if pain <= 4:
            score += 0.05
            contributions.append(ContributingFactor(
                "pain_score", 0.05,
                f"Controlled pain ({pain}/10) increases readiness",
                pain
            ))
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        return score, contributions[:5]
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(path))
        
        # Save scaler and metadata
        import joblib
        joblib.dump({
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_metadata": self.model_metadata,
            "is_fitted": self.is_fitted,
        }, str(path).replace(".json", "_metadata.joblib"))
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "DischargeReadinessModel":
        """Load model from disk."""
        import joblib
        
        instance = cls()
        instance.model.load_model(path)
        
        # Load metadata
        metadata_path = path.replace(".json", "_metadata.joblib")
        if Path(metadata_path).exists():
            metadata = joblib.load(metadata_path)
            instance.scaler = metadata["scaler"]
            instance.feature_names = metadata["feature_names"]
            instance.model_metadata = metadata["model_metadata"]
            instance.is_fitted = metadata["is_fitted"]
        else:
            instance.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return instance


# =============================================================================
# HYBRID DISCHARGE ANALYZER
# =============================================================================

class DischargeAnalyzer:
    """
    Hybrid ML + Rules discharge analysis system.
    
    This is the main class that combines the XGBoost readiness model with
    the clinical rule engine to produce explainable discharge recommendations.
    
    The hybrid approach ensures:
    1. ML model provides nuanced readiness scoring based on patterns
    2. Rule engine provides hard safety guarantees
    3. Rules can VETO ML recommendations for patient safety
    4. All decisions are explainable with specific reasons
    
    Example:
        >>> analyzer = DischargeAnalyzer()
        >>> patient_data = fetch_patient_clinical_data("P12345")
        >>> assessment = analyzer.analyze_patient(patient_data)
        >>> print(assessment.recommendation)
        "VETOED"
        >>> print(assessment.rule_vetoes[0].reason)
        "Temperature 38.7°C exceeds safe threshold (38.0°C)"
    """
    
    def __init__(self):
        """Initialize the hybrid analyzer."""
        self.rule_engine = ClinicalRuleEngine()
        self.ml_model = DischargeReadinessModel()
        
        # Try to load pre-trained model if available
        model_path = settings.model_path
        if Path(model_path).exists():
            try:
                self.ml_model = DischargeReadinessModel.load(model_path)
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
        
        logger.info("DischargeAnalyzer initialized with hybrid ML + Rules approach")
    
    def analyze_patient(
        self,
        patient_data: Dict[str, Any],
    ) -> DischargeAssessment:
        """
        Perform comprehensive discharge analysis for a patient.
        
        This method:
        1. Runs the ML model to get a readiness score
        2. Runs the rule engine to check for vetoes
        3. Combines results into final recommendation
        4. Generates explainability information
        
        Args:
            patient_data: Dictionary containing all patient clinical data
        
        Returns:
            DischargeAssessment with complete analysis results
        """
        patient_id = patient_data.get("patient_id", "unknown")
        
        logger.info(f"Analyzing discharge readiness for patient {patient_id}")
        
        # Step 1: Get ML readiness score
        ml_score, contributing_factors = self.ml_model.predict_readiness(patient_data)
        
        # Step 2: Run clinical rules
        rule_vetoes = self.rule_engine.evaluate(patient_data)
        
        # Step 3: Determine final recommendation
        recommendation = self._determine_recommendation(ml_score, rule_vetoes)
        
        # Step 4: Generate score interpretation
        score_interpretation = self._interpret_score(ml_score)
        
        # Calculate rule stats
        total_rules = 15  # Approximate total rules in engine
        rules_failed = len(rule_vetoes)
        rules_passed = total_rules - rules_failed
        
        # Calculate length of stay
        admission_dt = patient_data.get("admission_datetime")
        los_hours = None
        if admission_dt:
            los_hours = (datetime.utcnow() - admission_dt).total_seconds() / 3600
        
        # Build assessment
        assessment = DischargeAssessment(
            patient_id=patient_id,
            assessment_timestamp=datetime.utcnow(),
            recommendation=recommendation,
            ml_readiness_score=ml_score,
            ml_score_interpretation=score_interpretation,
            rule_vetoes=rule_vetoes,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
            contributing_factors=contributing_factors,
            length_of_stay_hours=los_hours,
            admission_type=patient_data.get("admission_type"),
            primary_diagnosis=patient_data.get("primary_diagnosis"),
            model_version=self.ml_model.model_metadata.get("trained_at", "synthetic"),
            rule_engine_version=ClinicalRuleEngine.VERSION,
        )
        
        logger.info(
            f"Analysis complete for {patient_id}: {recommendation.value} "
            f"(ML score: {ml_score:.2f}, vetoes: {rules_failed})"
        )
        
        return assessment
    
    def analyze_batch(
        self,
        patients_data: List[Dict[str, Any]],
    ) -> List[DischargeAssessment]:
        """
        Analyze multiple patients for discharge readiness.
        
        Args:
            patients_data: List of patient data dictionaries
        
        Returns:
            List of DischargeAssessment objects
        """
        logger.info(f"Batch analyzing {len(patients_data)} patients")
        
        assessments = []
        for patient_data in patients_data:
            try:
                assessment = self.analyze_patient(patient_data)
                assessments.append(assessment)
            except Exception as e:
                logger.error(
                    f"Error analyzing patient {patient_data.get('patient_id', 'unknown')}: {e}"
                )
                # Continue with other patients
        
        # Log summary
        ready = sum(1 for a in assessments if a.recommendation == DischargeRecommendation.LIKELY_READY)
        review = sum(1 for a in assessments if a.recommendation == DischargeRecommendation.NEEDS_REVIEW)
        not_ready = sum(1 for a in assessments if a.recommendation in [
            DischargeRecommendation.NOT_READY, DischargeRecommendation.VETOED
        ])
        
        logger.info(
            f"Batch analysis complete: {ready} likely ready, {review} need review, "
            f"{not_ready} not ready"
        )
        
        return assessments
    
    def _determine_recommendation(
        self,
        ml_score: float,
        vetoes: List[RuleVeto],
    ) -> DischargeRecommendation:
        """
        Determine final discharge recommendation.
        
        Critical logic:
        - ANY critical veto -> NOT_READY (rules override ML)
        - ANY warning veto -> NEEDS_REVIEW at best
        - High ML score with no vetoes -> LIKELY_READY
        - Medium ML score with no vetoes -> NEEDS_REVIEW
        - Low ML score -> NOT_READY
        """
        # Check for critical vetoes (these ALWAYS block discharge)
        critical_vetoes = [v for v in vetoes if v.severity == VetoSeverity.CRITICAL]
        if critical_vetoes:
            return DischargeRecommendation.VETOED
        
        # Check for warning vetoes
        warning_vetoes = [v for v in vetoes if v.severity == VetoSeverity.WARNING]
        
        # Apply ML score thresholds
        if ml_score >= settings.readiness_score_high_threshold:
            if warning_vetoes:
                return DischargeRecommendation.NEEDS_REVIEW
            return DischargeRecommendation.LIKELY_READY
        
        elif ml_score >= settings.readiness_score_medium_threshold:
            return DischargeRecommendation.NEEDS_REVIEW
        
        else:
            return DischargeRecommendation.NOT_READY
    
    def _interpret_score(self, score: float) -> str:
        """Generate human-readable interpretation of ML score."""
        if score >= 0.8:
            return (
                "High readiness score indicates patient's clinical trajectory "
                "is consistent with successful discharge patterns."
            )
        elif score >= 0.6:
            return (
                "Moderate-high readiness score suggests patient is approaching "
                "discharge readiness. Review clinical rules for remaining concerns."
            )
        elif score >= 0.4:
            return (
                "Moderate readiness score indicates patient may need additional "
                "stabilization or intervention before discharge is appropriate."
            )
        elif score >= 0.2:
            return (
                "Low-moderate readiness score suggests patient is not yet ready "
                "for discharge. Multiple clinical factors need improvement."
            )
        else:
            return (
                "Low readiness score indicates patient requires continued "
                "inpatient care. Focus on acute medical management."
            )
