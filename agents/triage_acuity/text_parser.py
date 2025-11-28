"""
Triage & Acuity Agent - Text Parser (Red Flag NLP Module)

This module implements the critical safety mechanism for Emergency Medicine triage:
automatic detection of "Red Flag" symptoms in free-text patient complaints.

================================================================================
RED FLAG LOGIC: A FAIL-SAFE MECHANISM FOR EMERGENCY MEDICINE
================================================================================

WHAT ARE RED FLAGS?
───────────────────
Red Flags are symptoms or presentations that indicate potentially life-threatening
conditions requiring IMMEDIATE medical attention. When detected, they OVERRIDE
the machine learning model's prediction to ensure patient safety.

WHY OVERRIDE ML PREDICTIONS?
────────────────────────────
1. SAFETY FIRST: No ML model is perfect. False negatives (missing a critical 
   patient) can be fatal. It's better to over-triage than under-triage.

2. MEDICOLEGAL PROTECTION: Documenting that Red Flags triggered escalation
   provides clinical audit trail and reduces liability.

3. CLINICAL STANDARD OF CARE: Red Flag detection mirrors how human nurses
   are trained - certain symptoms always warrant immediate attention.

4. MODEL LIMITATIONS: ML models learn from historical data. Novel presentations
   of critical conditions may not be well-represented in training data.

RED FLAG CATEGORIES:
────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│  CATEGORY          │  EXAMPLES                  │  RATIONALE               │
├─────────────────────────────────────────────────────────────────────────────┤
│  CARDIAC           │  chest pain, crushing      │  MI, PE, Aortic          │
│                    │  pressure, heart attack    │  dissection              │
├─────────────────────────────────────────────────────────────────────────────┤
│  RESPIRATORY       │  can't breathe, difficulty │  Airway compromise,      │
│                    │  breathing, choking        │  anaphylaxis, asthma     │
├─────────────────────────────────────────────────────────────────────────────┤
│  NEUROLOGICAL      │  stroke, weakness one side,│  CVA, TIA, hemorrhage    │
│                    │  slurred speech, seizure   │                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ALTERED MENTAL    │  unresponsive, confused,   │  Shock, overdose,        │
│  STATUS            │  unconscious, passing out  │  sepsis, DKA             │
├─────────────────────────────────────────────────────────────────────────────┤
│  SEVERE BLEEDING   │  won't stop bleeding,      │  Hemorrhage, trauma      │
│                    │  vomiting blood            │                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ANAPHYLAXIS       │  throat closing, severe    │  Allergic emergency      │
│                    │  allergic reaction, hives  │                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  OBSTETRIC         │  labor, water broke,       │  Emergency delivery      │
│                    │  contractions, pregnancy   │                          │
│                    │  bleeding                  │                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  TRAUMA            │  gunshot, stabbing, car    │  Major trauma protocol   │
│                    │  accident, severe injury   │                          │
└─────────────────────────────────────────────────────────────────────────────┘

IMPLEMENTATION NOTES:
─────────────────────
- Uses regex for speed and simplicity (no external NLP dependencies)
- Case-insensitive matching
- Supports common misspellings and abbreviations
- Returns confidence scores based on keyword specificity
- Audit logging for all detections

================================================================================
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from config import settings, AcuityLevel


# Configure logging
logger = logging.getLogger(__name__)


class RedFlagCategory(Enum):
    """Categories of Red Flag symptoms."""
    CARDIAC = "cardiac"
    RESPIRATORY = "respiratory"
    NEUROLOGICAL = "neurological"
    ALTERED_MENTAL_STATUS = "altered_mental_status"
    SEVERE_BLEEDING = "severe_bleeding"
    ANAPHYLAXIS = "anaphylaxis"
    OBSTETRIC = "obstetric"
    TRAUMA = "trauma"
    SEPSIS = "sepsis"
    PEDIATRIC = "pediatric"


@dataclass
class RedFlagMatch:
    """
    Represents a detected Red Flag in patient text.
    
    Attributes:
        keyword: The matched keyword/phrase
        category: The Red Flag category
        severity: Severity score (1.0 = highest)
        matched_text: The actual text that matched
        position: Character position in original text
        suggested_acuity: Recommended acuity level (1 or 2)
    """
    keyword: str
    category: RedFlagCategory
    severity: float
    matched_text: str
    position: int
    suggested_acuity: int = 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "keyword": self.keyword,
            "category": self.category.value,
            "severity": self.severity,
            "matched_text": self.matched_text,
            "position": self.position,
            "suggested_acuity": self.suggested_acuity,
        }


@dataclass
class ParseResult:
    """
    Result of parsing patient symptoms text.
    
    Attributes:
        original_text: The input text
        red_flags: List of detected Red Flags
        has_critical_flags: Whether any critical (level 1) flags were found
        override_acuity: Suggested acuity override (None if no Red Flags)
        confidence: Confidence in the Red Flag detection (0-1)
        audit_message: Human-readable summary for clinical audit
    """
    original_text: str
    red_flags: List[RedFlagMatch] = field(default_factory=list)
    has_critical_flags: bool = False
    override_acuity: Optional[int] = None
    confidence: float = 0.0
    audit_message: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "original_text": self.original_text,
            "red_flags": [rf.to_dict() for rf in self.red_flags],
            "has_critical_flags": self.has_critical_flags,
            "override_acuity": self.override_acuity,
            "confidence": self.confidence,
            "audit_message": self.audit_message,
        }


class RedFlagParser:
    """
    Parser for detecting Red Flag symptoms in patient complaint text.
    
    This is the core safety mechanism that ensures critical patients
    are never under-triaged by the ML model.
    
    Usage:
        parser = RedFlagParser()
        result = parser.parse("Patient complains of severe chest pain and difficulty breathing")
        if result.has_critical_flags:
            acuity = result.override_acuity  # Will be 1 or 2
    """
    
    # ==========================================================================
    # RED FLAG KEYWORD DEFINITIONS
    # ==========================================================================
    # Each tuple: (keyword/pattern, severity, suggested_acuity)
    # Severity: 1.0 = definite critical, 0.7 = likely critical, 0.5 = possible
    
    RED_FLAG_PATTERNS: Dict[RedFlagCategory, List[Tuple[str, float, int]]] = {
        
        RedFlagCategory.CARDIAC: [
            # Definite cardiac emergencies
            (r"\bchest\s*pain\b", 1.0, 1),
            (r"\bheart\s*attack\b", 1.0, 1),
            (r"\bcardiac\s*arrest\b", 1.0, 1),
            (r"\bcrushing\s*(chest\s*)?(pain|pressure)\b", 1.0, 1),
            (r"\bpressure\s*(in|on)\s*(my\s*)?chest\b", 1.0, 1),
            (r"\btight(ness)?\s*(in|on)\s*(my\s*)?chest\b", 0.9, 1),
            (r"\bradiating\s*(to\s*)?(arm|jaw|back)\b", 0.9, 1),
            (r"\bpalpitations?\b", 0.6, 2),
            (r"\birregular\s*heart\s*beat\b", 0.7, 2),
            (r"\bheart\s*(is\s*)?(racing|pounding)\b", 0.6, 2),
            (r"\bcp\b", 0.8, 1),  # Common abbreviation for chest pain
        ],
        
        RedFlagCategory.RESPIRATORY: [
            # Airway emergencies
            (r"\b(can'?t|cannot|couldn'?t|unable\s*to)\s*breathe?\b", 1.0, 1),
            (r"\bdifficulty\s*breathing\b", 0.9, 1),
            (r"\bshortness\s*of\s*breath\b", 0.8, 2),
            (r"\bsob\b", 0.8, 2),  # Shortness of breath abbreviation
            (r"\bbreathing\s*(problem|trouble|difficulty)\b", 0.8, 2),
            (r"\bchoking\b", 1.0, 1),
            (r"\bairway\s*(blocked?|obstruct)\b", 1.0, 1),
            (r"\bsuffocating\b", 1.0, 1),
            (r"\bgasping\s*(for\s*)?(air|breath)\b", 1.0, 1),
            (r"\bwheez(ing|e)\b", 0.6, 2),
            (r"\basthma\s*attack\b", 0.8, 2),
            (r"\bstridor\b", 0.9, 1),
            (r"\bcyanosis\b", 1.0, 1),
            (r"\bblue\s*(lips?|face|skin)\b", 1.0, 1),
            (r"\brespiratory\s*distress\b", 0.9, 1),
        ],
        
        RedFlagCategory.NEUROLOGICAL: [
            # Stroke and neurological emergencies
            (r"\bstroke\b", 1.0, 1),
            (r"\bcva\b", 1.0, 1),  # Cerebrovascular accident
            (r"\btia\b", 0.9, 1),  # Transient ischemic attack
            (r"\bslurred\s*speech\b", 0.9, 1),
            (r"\bfacial\s*(droop(ing)?|weakness)\b", 0.9, 1),
            (r"\bweakness\s*(on\s*)?(one|left|right)\s*side\b", 0.9, 1),
            (r"\b(arm|leg)\s*weakness\b", 0.7, 2),
            (r"\bnumb(ness)?\s*(on\s*)?(one|left|right)\s*side\b", 0.8, 2),
            (r"\bseizure\b", 0.9, 1),
            (r"\bconvulsion\b", 0.9, 1),
            (r"\bfit\b", 0.6, 2),
            (r"\b(severe|worst)\s*headache\b", 0.8, 2),
            (r"\bthunderclap\s*headache\b", 1.0, 1),  # Subarachnoid hemorrhage
            (r"\bsudden\s*(onset\s*)?headache\b", 0.8, 2),
            (r"\bvision\s*(loss|changes?|problem)\b", 0.7, 2),
            (r"\bdouble\s*vision\b", 0.7, 2),
            (r"\bsudden\s*blindness\b", 0.9, 1),
        ],
        
        RedFlagCategory.ALTERED_MENTAL_STATUS: [
            # Consciousness and mental status
            (r"\bunresponsive\b", 1.0, 1),
            (r"\bunconscious\b", 1.0, 1),
            (r"\bnot\s*(responding|waking)\b", 1.0, 1),
            (r"\bpassed?\s*out\b", 0.8, 2),
            (r"\bfainted?\b", 0.7, 2),
            (r"\bsyncope\b", 0.7, 2),
            (r"\bconfused?\b", 0.6, 2),
            (r"\bdisoriented\b", 0.7, 2),
            (r"\blethar(gic|gy)\b", 0.7, 2),
            (r"\baltered\s*(mental\s*)?(status|consciousness)\b", 0.8, 2),
            (r"\bams\b", 0.8, 2),  # Altered mental status
            (r"\bloss\s*of\s*consciousness\b", 0.9, 1),
            (r"\bloc\b", 0.9, 1),  # Loss of consciousness
            (r"\bdrowsy\b", 0.5, 3),
            (r"\bwon'?t\s*wake\s*up\b", 1.0, 1),
            (r"\bunrousable\b", 1.0, 1),
        ],
        
        RedFlagCategory.SEVERE_BLEEDING: [
            # Hemorrhage
            (r"\b(won'?t|can'?t)\s*stop\s*bleeding\b", 0.9, 1),
            (r"\bsevere\s*bleed(ing)?\b", 0.9, 1),
            (r"\bhemorrhag(e|ing)\b", 0.9, 1),
            (r"\bvomiting\s*blood\b", 0.9, 1),
            (r"\bhematemesis\b", 0.9, 1),
            (r"\bcoughing\s*(up\s*)?blood\b", 0.8, 2),
            (r"\bhemoptysis\b", 0.8, 2),
            (r"\bblood\s*in\s*(stool|urine)\b", 0.7, 2),
            (r"\bmelena\b", 0.8, 2),
            (r"\bhematuria\b", 0.6, 2),
            (r"\brectal\s*bleed(ing)?\b", 0.7, 2),
            (r"\buncontrolled\s*bleed(ing)?\b", 0.9, 1),
            (r"\barteri(al|y)\s*bleed(ing)?\b", 0.9, 1),
        ],
        
        RedFlagCategory.ANAPHYLAXIS: [
            # Allergic emergencies
            (r"\bthroat\s*(closing|swelling|tight)\b", 1.0, 1),
            (r"\banaphyla(xis|ctic)\b", 1.0, 1),
            (r"\bsevere\s*allergic\s*reaction\b", 1.0, 1),
            (r"\ballergic\s*reaction\b", 0.7, 2),
            (r"\bhives\s*(all\s*over|everywhere)\b", 0.7, 2),
            (r"\bface\s*(swelling|swollen)\b", 0.8, 2),
            (r"\bangioedema\b", 0.9, 1),
            (r"\blips?\s*(swelling|swollen)\b", 0.8, 2),
            (r"\btongue\s*(swelling|swollen)\b", 0.9, 1),
            (r"\bepipen\b", 0.7, 2),  # If mentioning epipen, likely allergic emergency
            (r"\bbee\s*sting\b.*\b(allergic|reaction|swelling)\b", 0.8, 2),
        ],
        
        RedFlagCategory.OBSTETRIC: [
            # Pregnancy emergencies
            (r"\bin\s*labor\b", 0.9, 1),
            (r"\bwater\s*broke\b", 0.9, 1),
            (r"\bcontractions?\b", 0.7, 2),
            (r"\bpregnant\b.*\b(bleed|pain|water)\b", 0.8, 2),
            (r"\bvaginal\s*bleed(ing)?\b.*\bpregnant\b", 0.9, 1),
            (r"\bectopic\b", 0.9, 1),
            (r"\bmiscarriage\b", 0.8, 2),
            (r"\bbaby\s*(coming|crowning)\b", 1.0, 1),
            (r"\bumbilical\s*cord\b", 1.0, 1),
            (r"\bplacent(a|al)\b", 0.8, 2),
            (r"\bpreeclampsia\b", 0.9, 1),
            (r"\beclampsia\b", 1.0, 1),
        ],
        
        RedFlagCategory.TRAUMA: [
            # Major trauma
            (r"\bgunshot\b", 1.0, 1),
            (r"\bgsw\b", 1.0, 1),  # Gunshot wound
            (r"\bstab(bed|bing|wound)\b", 1.0, 1),
            (r"\bcar\s*(accident|crash|wreck)\b", 0.8, 2),
            (r"\bmva\b", 0.8, 2),  # Motor vehicle accident
            (r"\bmvc\b", 0.8, 2),  # Motor vehicle collision
            (r"\bpedestrian\s*(struck|hit)\b", 0.9, 1),
            (r"\bfall\s*from\s*(height|ladder|roof)\b", 0.8, 2),
            (r"\bhead\s*(injury|trauma)\b", 0.8, 2),
            (r"\bsevere\s*(injury|trauma)\b", 0.8, 2),
            (r"\bamputation\b", 0.9, 1),
            (r"\bcrushing\s*(injury)?\b", 0.9, 1),
            (r"\bburns?\s*(over|to)\s*\d+\s*%\b", 0.9, 1),
            (r"\binhalation\s*injury\b", 0.9, 1),
            (r"\belectrocution\b", 0.9, 1),
            (r"\bdrowning\b", 1.0, 1),
            (r"\bhanging\b", 1.0, 1),
        ],
        
        RedFlagCategory.SEPSIS: [
            # Infection and sepsis
            (r"\bsepsis\b", 0.9, 1),
            (r"\bseptic\b", 0.9, 1),
            (r"\bhigh\s*fever\b.*\b(confused|lethargy|weak)\b", 0.8, 2),
            (r"\bfever\b.*\b(rash|stiff\s*neck)\b", 0.8, 2),
            (r"\bmeningitis\b", 0.9, 1),
            (r"\bnecrotizing\b", 0.9, 1),
            (r"\bflesh\s*eating\b", 0.9, 1),
            (r"\bimmunocompromised\b.*\bfever\b", 0.8, 2),
            (r"\bneutropenic\s*fever\b", 0.9, 1),
        ],
        
        RedFlagCategory.PEDIATRIC: [
            # Pediatric-specific emergencies
            (r"\b(infant|baby|newborn)\b.*\b(not\s*breathing|blue|limp)\b", 1.0, 1),
            (r"\b(child|toddler)\b.*\bunresponsive\b", 1.0, 1),
            (r"\bsids\b", 1.0, 1),
            (r"\bshaken\s*baby\b", 1.0, 1),
            (r"\bbaby\b.*\b(fever|temp)\b.*\b(under|less\s*than)\s*(3|three)\s*months?\b", 0.9, 1),
            (r"\bfebrile\s*seizure\b", 0.8, 2),
            (r"\b(child|baby)\b.*\bchoking\b", 1.0, 1),
            (r"\bingestion\b", 0.7, 2),
            (r"\bpoisoning\b", 0.8, 2),
            (r"\boverdose\b", 0.9, 1),
        ],
    }
    
    def __init__(self):
        """Initialize the Red Flag parser with compiled regex patterns."""
        self.compiled_patterns: Dict[RedFlagCategory, List[Tuple[re.Pattern, float, int]]] = {}
        
        for category, patterns in self.RED_FLAG_PATTERNS.items():
            self.compiled_patterns[category] = [
                (re.compile(pattern, re.IGNORECASE), severity, acuity)
                for pattern, severity, acuity in patterns
            ]
        
        logger.info(
            "RedFlagParser initialized",
            extra={
                "categories": len(self.compiled_patterns),
                "total_patterns": sum(len(p) for p in self.compiled_patterns.values()),
            }
        )
    
    def parse(self, text: str) -> ParseResult:
        """
        Parse patient symptoms text for Red Flag keywords.
        
        Args:
            text: Free-text patient complaint or symptoms description
        
        Returns:
            ParseResult containing detected Red Flags and override recommendations
        """
        if not text or not text.strip():
            return ParseResult(
                original_text=text or "",
                audit_message="Empty text provided, no Red Flags to detect",
            )
        
        text = text.strip()
        red_flags: List[RedFlagMatch] = []
        
        # Search all categories
        for category, patterns in self.compiled_patterns.items():
            for pattern, severity, suggested_acuity in patterns:
                for match in pattern.finditer(text):
                    red_flag = RedFlagMatch(
                        keyword=pattern.pattern,
                        category=category,
                        severity=severity,
                        matched_text=match.group(),
                        position=match.start(),
                        suggested_acuity=suggested_acuity,
                    )
                    red_flags.append(red_flag)
        
        # Deduplicate overlapping matches (keep highest severity)
        red_flags = self._deduplicate_matches(red_flags)
        
        # Sort by severity (highest first)
        red_flags.sort(key=lambda x: (-x.severity, x.position))
        
        # Determine override acuity
        has_critical = any(rf.suggested_acuity == 1 for rf in red_flags)
        override_acuity = None
        confidence = 0.0
        
        if red_flags:
            # Use the most severe suggested acuity
            override_acuity = min(rf.suggested_acuity for rf in red_flags)
            # Confidence is based on highest severity match
            confidence = max(rf.severity for rf in red_flags)
        
        # Generate audit message
        audit_message = self._generate_audit_message(red_flags, override_acuity)
        
        result = ParseResult(
            original_text=text,
            red_flags=red_flags,
            has_critical_flags=has_critical,
            override_acuity=override_acuity if settings.red_flag_override_enabled else None,
            confidence=confidence,
            audit_message=audit_message,
        )
        
        # Audit logging
        if red_flags and settings.red_flag_audit_logging:
            logger.warning(
                "RED FLAG DETECTED",
                extra={
                    "flags_count": len(red_flags),
                    "categories": list(set(rf.category.value for rf in red_flags)),
                    "override_acuity": override_acuity,
                    "confidence": confidence,
                    "matched_keywords": [rf.matched_text for rf in red_flags[:5]],
                }
            )
        
        return result
    
    def _deduplicate_matches(self, matches: List[RedFlagMatch]) -> List[RedFlagMatch]:
        """
        Remove overlapping matches, keeping highest severity.
        
        If two patterns match overlapping text, keep the more specific/severe one.
        """
        if not matches:
            return []
        
        # Sort by position, then by severity (descending)
        sorted_matches = sorted(matches, key=lambda x: (x.position, -x.severity))
        
        result = []
        last_end = -1
        
        for match in sorted_matches:
            match_end = match.position + len(match.matched_text)
            
            # Check for overlap with previous match
            if match.position >= last_end:
                result.append(match)
                last_end = match_end
            elif match.severity > result[-1].severity:
                # Replace previous with higher severity
                result[-1] = match
                last_end = match_end
        
        return result
    
    def _generate_audit_message(
        self,
        red_flags: List[RedFlagMatch],
        override_acuity: Optional[int],
    ) -> str:
        """Generate human-readable audit message for clinical documentation."""
        if not red_flags:
            return "No Red Flag symptoms detected in patient complaint."
        
        categories = list(set(rf.category.value for rf in red_flags))
        keywords = [rf.matched_text for rf in red_flags[:3]]
        
        message_parts = [
            f"RED FLAG ALERT: {len(red_flags)} potential critical symptom(s) detected.",
            f"Categories: {', '.join(categories)}.",
            f"Key matches: {', '.join(keywords)}.",
        ]
        
        if override_acuity:
            acuity_label = AcuityLevel.get_label(override_acuity)
            message_parts.append(
                f"RECOMMENDATION: Override to Acuity {override_acuity} ({acuity_label})."
            )
        
        return " ".join(message_parts)
    
    def get_category_patterns(self, category: RedFlagCategory) -> List[str]:
        """Get list of pattern keywords for a specific category."""
        return [
            pattern.pattern
            for pattern, _, _ in self.compiled_patterns.get(category, [])
        ]
    
    def add_custom_pattern(
        self,
        category: RedFlagCategory,
        pattern: str,
        severity: float,
        suggested_acuity: int,
    ) -> None:
        """
        Add a custom Red Flag pattern dynamically.
        
        This allows hospitals to add institution-specific patterns
        without modifying the code.
        """
        if category not in self.compiled_patterns:
            self.compiled_patterns[category] = []
        
        compiled = re.compile(pattern, re.IGNORECASE)
        self.compiled_patterns[category].append((compiled, severity, suggested_acuity))
        
        logger.info(
            f"Added custom Red Flag pattern",
            extra={
                "category": category.value,
                "pattern": pattern,
                "severity": severity,
            }
        )


# =============================================================================
# VITAL SIGNS PARSER
# =============================================================================

@dataclass
class VitalSigns:
    """
    Structured vital signs data.
    
    Normal ranges for adult patients:
    - Heart Rate: 60-100 bpm
    - Systolic BP: 90-140 mmHg
    - Diastolic BP: 60-90 mmHg
    - Respiratory Rate: 12-20 breaths/min
    - SpO2: 95-100%
    - Temperature: 36.1-37.2°C (97-99°F)
    - GCS: 15 (normal), <9 (severe impairment)
    """
    heart_rate: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    respiratory_rate: Optional[float] = None
    oxygen_saturation: Optional[float] = None  # SpO2
    temperature: Optional[float] = None  # Celsius
    gcs: Optional[int] = None  # Glasgow Coma Scale (3-15)
    
    def check_critical_vitals(self) -> List[str]:
        """
        Check for critically abnormal vital signs.
        
        Returns list of critical findings that should trigger acuity override.
        """
        critical_findings = []
        
        # Heart rate checks
        if self.heart_rate is not None:
            if self.heart_rate < 40:
                critical_findings.append(f"Severe bradycardia (HR: {self.heart_rate})")
            elif self.heart_rate > 150:
                critical_findings.append(f"Severe tachycardia (HR: {self.heart_rate})")
        
        # Blood pressure checks
        if self.systolic_bp is not None:
            if self.systolic_bp < 90:
                critical_findings.append(f"Hypotension (SBP: {self.systolic_bp})")
            elif self.systolic_bp > 180:
                critical_findings.append(f"Hypertensive crisis (SBP: {self.systolic_bp})")
        
        # Respiratory rate checks
        if self.respiratory_rate is not None:
            if self.respiratory_rate < 8:
                critical_findings.append(f"Respiratory depression (RR: {self.respiratory_rate})")
            elif self.respiratory_rate > 30:
                critical_findings.append(f"Respiratory distress (RR: {self.respiratory_rate})")
        
        # Oxygen saturation checks
        if self.oxygen_saturation is not None:
            if self.oxygen_saturation < 90:
                critical_findings.append(f"Hypoxia (SpO2: {self.oxygen_saturation}%)")
        
        # Temperature checks
        if self.temperature is not None:
            if self.temperature > 40.0:
                critical_findings.append(f"Hyperthermia (Temp: {self.temperature}°C)")
            elif self.temperature < 35.0:
                critical_findings.append(f"Hypothermia (Temp: {self.temperature}°C)")
        
        # GCS checks
        if self.gcs is not None:
            if self.gcs <= 8:
                critical_findings.append(f"Severe impairment (GCS: {self.gcs})")
        
        return critical_findings


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

# Create a singleton parser instance for reuse
_parser_instance: Optional[RedFlagParser] = None


def get_parser() -> RedFlagParser:
    """Get or create the singleton RedFlagParser instance."""
    global _parser_instance
    if _parser_instance is None:
        _parser_instance = RedFlagParser()
    return _parser_instance


def parse_symptoms(text: str) -> ParseResult:
    """
    Convenience function to parse symptoms text.
    
    Args:
        text: Patient complaint or symptoms description
    
    Returns:
        ParseResult with detected Red Flags
    """
    return get_parser().parse(text)
