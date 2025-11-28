"""
Configuration module for ER/OR Scheduling Agent.

Manages environment variables and default settings for:
- Server configuration
- ML model paths
- OR scheduling parameters
- Acuity priority weights
"""

import os
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path


@dataclass
class ServerConfig:
    """Server configuration settings."""
    host: str = "0.0.0.0"
    port: int = 8003
    debug: bool = False


@dataclass
class ModelConfig:
    """ML model configuration."""
    model_path: Path = field(default_factory=lambda: Path("./models/duration_predictor.json"))
    retrain_on_startup: bool = True


@dataclass
class SchedulingConfig:
    """OR scheduling parameters."""
    default_or_count: int = 4
    default_block_duration_minutes: int = 480  # 8-hour blocks
    max_solver_time_seconds: int = 30
    scheduling_horizon_days: int = 7
    surgery_buffer_minutes: int = 30
    cleanup_time_minutes: int = 15


@dataclass
class AcuityConfig:
    """Patient acuity priority weights."""
    weights: Dict[str, int] = field(default_factory=lambda: {
        "critical": 100,
        "emergent": 75,
        "urgent": 50,
        "less_urgent": 25,
        "non_urgent": 10
    })
    
    # ESI (Emergency Severity Index) mapping
    esi_mapping: Dict[int, str] = field(default_factory=lambda: {
        1: "critical",
        2: "emergent",
        3: "urgent",
        4: "less_urgent",
        5: "non_urgent"
    })


@dataclass
class Config:
    """Main configuration container."""
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    acuity: AcuityConfig = field(default_factory=AcuityConfig)
    orchestrator_url: str = "http://orchestrator:3000"
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        server = ServerConfig(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8003")),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )
        
        model = ModelConfig(
            model_path=Path(os.getenv("MODEL_PATH", "./models/duration_predictor.json")),
            retrain_on_startup=os.getenv("RETRAIN_ON_STARTUP", "true").lower() == "true"
        )
        
        scheduling = SchedulingConfig(
            default_or_count=int(os.getenv("DEFAULT_OR_COUNT", "4")),
            default_block_duration_minutes=int(os.getenv("DEFAULT_BLOCK_DURATION_MINUTES", "480")),
            max_solver_time_seconds=int(os.getenv("MAX_SOLVER_TIME_SECONDS", "30")),
            scheduling_horizon_days=int(os.getenv("SCHEDULING_HORIZON_DAYS", "7")),
            surgery_buffer_minutes=int(os.getenv("SURGERY_BUFFER_MINUTES", "30")),
            cleanup_time_minutes=int(os.getenv("CLEANUP_TIME_MINUTES", "15"))
        )
        
        acuity = AcuityConfig(
            weights={
                "critical": int(os.getenv("ACUITY_WEIGHT_CRITICAL", "100")),
                "emergent": int(os.getenv("ACUITY_WEIGHT_EMERGENT", "75")),
                "urgent": int(os.getenv("ACUITY_WEIGHT_URGENT", "50")),
                "less_urgent": int(os.getenv("ACUITY_WEIGHT_LESS_URGENT", "25")),
                "non_urgent": int(os.getenv("ACUITY_WEIGHT_NON_URGENT", "10"))
            }
        )
        
        return cls(
            server=server,
            model=model,
            scheduling=scheduling,
            acuity=acuity,
            orchestrator_url=os.getenv("ORCHESTRATOR_URL", "http://orchestrator:3000"),
            log_level=os.getenv("LOG_LEVEL", "INFO")
        )


# Global configuration instance
config = Config.from_env()


# Procedure type definitions with base durations (minutes)
PROCEDURE_TYPES = {
    "appendectomy": {"base_duration": 60, "complexity": 1},
    "cholecystectomy": {"base_duration": 90, "complexity": 2},
    "hernia_repair": {"base_duration": 75, "complexity": 1},
    "knee_arthroscopy": {"base_duration": 45, "complexity": 1},
    "hip_replacement": {"base_duration": 180, "complexity": 3},
    "knee_replacement": {"base_duration": 150, "complexity": 3},
    "cardiac_bypass": {"base_duration": 300, "complexity": 5},
    "spinal_fusion": {"base_duration": 240, "complexity": 4},
    "tumor_resection": {"base_duration": 180, "complexity": 4},
    "emergency_laparotomy": {"base_duration": 120, "complexity": 3},
    "cesarean_section": {"base_duration": 60, "complexity": 2},
    "craniotomy": {"base_duration": 270, "complexity": 5},
    "mastectomy": {"base_duration": 120, "complexity": 2},
    "colectomy": {"base_duration": 180, "complexity": 3},
    "thyroidectomy": {"base_duration": 90, "complexity": 2},
}

# OR Room definitions
OR_ROOMS = {
    "OR-1": {"specialty": ["general", "emergency"], "equipment": ["standard", "laparoscopic"]},
    "OR-2": {"specialty": ["orthopedic"], "equipment": ["standard", "arthroscopic", "implants"]},
    "OR-3": {"specialty": ["cardiac", "thoracic"], "equipment": ["bypass_machine", "standard"]},
    "OR-4": {"specialty": ["neuro", "spine"], "equipment": ["microscope", "navigation", "standard"]},
}
