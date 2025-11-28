"""
OR Scheduling Module using Google OR-Tools.

Implements a Bin Packing approach to Operating Room scheduling:
- Each OR time block is a "bin" with fixed capacity (minutes)
- Each surgery is an "item" with predicted duration
- Goal: Pack surgeries into OR blocks minimizing idle time
  while respecting priority constraints

This demonstrates the optimization component of our Hybrid Agent.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

from ortools.sat.python import cp_model

from config import config, OR_ROOMS, PROCEDURE_TYPES
from duration_predictor import DurationPredictor, PredictionInput, PredictionResult

logger = logging.getLogger(__name__)


class SchedulingStatus(Enum):
    """Status of scheduling optimization."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class Surgery:
    """Represents a surgery to be scheduled."""
    surgery_id: str
    patient_id: str
    procedure_type: str
    patient_acuity: int  # ESI 1-5 (1 = most urgent)
    patient_age: int = 50
    patient_bmi: float = 25.0
    comorbidity_count: int = 0
    surgeon_id: Optional[str] = None
    surgeon_experience_years: int = 10
    is_emergency: bool = False
    required_equipment: List[str] = field(default_factory=list)
    specialty: str = "general"
    
    # Computed fields
    predicted_duration: int = 0
    priority_score: float = 0.0


@dataclass
class ORBlock:
    """Represents an OR time block (a "bin" in our packing problem)."""
    block_id: str
    room_id: str
    start_time: datetime
    end_time: datetime
    capacity_minutes: int
    specialty: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    
    @property
    def available_minutes(self) -> int:
        """Calculate total available minutes in block."""
        return int((self.end_time - self.start_time).total_seconds() / 60)


@dataclass
class ScheduledSurgery:
    """A surgery assigned to a specific OR block and time."""
    surgery: Surgery
    block_id: str
    room_id: str
    scheduled_start: datetime
    scheduled_end: datetime
    buffer_after: int  # Cleanup/turnover time


@dataclass
class ScheduleResult:
    """Result of the scheduling optimization."""
    status: SchedulingStatus
    scheduled_surgeries: List[ScheduledSurgery]
    unscheduled_surgeries: List[Surgery]
    total_idle_minutes: int
    utilization_percent: float
    solver_time_seconds: float
    objective_value: float


class ORScheduler:
    """
    Operating Room Scheduler using Constraint Programming.
    
    Formulates OR scheduling as a Bin Packing problem:
    - Bins = OR time blocks (fixed capacity in minutes)
    - Items = Surgeries (with predicted durations)
    - Objective = Minimize total idle time while prioritizing urgent cases
    
    The scheduler integrates ML predictions (surgery duration) into the
    optimization constraints, demonstrating a Hybrid Agent architecture.
    """
    
    def __init__(self, duration_predictor: Optional[DurationPredictor] = None):
        """Initialize the scheduler with optional duration predictor."""
        self.duration_predictor = duration_predictor
        self.config = config.scheduling
        self.acuity_weights = config.acuity.weights
    
    def calculate_priority_score(self, surgery: Surgery) -> float:
        """
        Calculate priority score for a surgery.
        
        Higher scores = higher priority.
        Combines acuity, emergency status, and wait time factors.
        """
        # Base score from acuity
        acuity_name = config.acuity.esi_mapping.get(surgery.patient_acuity, "urgent")
        base_score = self.acuity_weights.get(acuity_name, 50)
        
        # Emergency bonus
        emergency_bonus = 50 if surgery.is_emergency else 0
        
        # Complexity factor (more complex = schedule sooner to avoid delays)
        proc_info = PROCEDURE_TYPES.get(surgery.procedure_type, {"complexity": 2})
        complexity_factor = proc_info["complexity"] * 5
        
        return base_score + emergency_bonus + complexity_factor
    
    def predict_duration(self, surgery: Surgery) -> int:
        """
        Get predicted duration for a surgery.
        
        Uses ML predictor if available, otherwise falls back to heuristics.
        """
        if self.duration_predictor:
            pred_input = PredictionInput(
                procedure_type=surgery.procedure_type,
                patient_acuity=surgery.patient_acuity,
                patient_age=surgery.patient_age,
                patient_bmi=surgery.patient_bmi,
                comorbidity_count=surgery.comorbidity_count,
                surgeon_experience_years=surgery.surgeon_experience_years,
                is_emergency=surgery.is_emergency
            )
            result = self.duration_predictor.predict(pred_input)
            return result.predicted_duration_minutes
        
        # Fallback: use base duration from procedure type
        proc_info = PROCEDURE_TYPES.get(surgery.procedure_type, {"base_duration": 90})
        return proc_info["base_duration"]
    
    def prepare_surgeries(self, surgeries: List[Surgery]) -> List[Surgery]:
        """Prepare surgeries by computing durations and priority scores."""
        prepared = []
        for surgery in surgeries:
            surgery.predicted_duration = self.predict_duration(surgery)
            surgery.priority_score = self.calculate_priority_score(surgery)
            prepared.append(surgery)
        return prepared
    
    def generate_or_blocks(
        self,
        start_date: datetime,
        num_days: int = 1,
        rooms: Optional[Dict[str, Dict]] = None
    ) -> List[ORBlock]:
        """
        Generate OR blocks for scheduling.
        
        Creates time blocks for each OR room over the specified period.
        """
        rooms = rooms or OR_ROOMS
        blocks = []
        block_duration = self.config.default_block_duration_minutes
        
        for day_offset in range(num_days):
            current_date = start_date + timedelta(days=day_offset)
            
            for room_id, room_info in rooms.items():
                # Morning block (8 AM - 4 PM typically)
                block_start = current_date.replace(hour=8, minute=0, second=0, microsecond=0)
                block_end = block_start + timedelta(minutes=block_duration)
                
                blocks.append(ORBlock(
                    block_id=f"{room_id}_{current_date.strftime('%Y%m%d')}_AM",
                    room_id=room_id,
                    start_time=block_start,
                    end_time=block_end,
                    capacity_minutes=block_duration,
                    specialty=room_info.get("specialty", []),
                    equipment=room_info.get("equipment", [])
                ))
        
        return blocks
    
    def check_compatibility(self, surgery: Surgery, block: ORBlock) -> bool:
        """Check if a surgery can be scheduled in a specific OR block."""
        # Check specialty compatibility
        if surgery.specialty and block.specialty:
            if surgery.specialty not in block.specialty and "general" not in block.specialty:
                return False
        
        # Check equipment requirements
        for equip in surgery.required_equipment:
            if equip not in block.equipment and "standard" not in block.equipment:
                return False
        
        return True
    
    def solve_bin_packing(
        self,
        surgeries: List[Surgery],
        blocks: List[ORBlock]
    ) -> ScheduleResult:
        """
        Solve the OR scheduling problem using Constraint Programming.
        
        This is formulated as a variant of the Bin Packing problem:
        - Each OR block is a "bin" with capacity = available minutes
        - Each surgery is an "item" with size = predicted duration + buffer
        - We want to minimize "wasted space" (idle time) in bins
        - Priority constraints ensure urgent surgeries are scheduled first
        
        Decision Variables:
        - x[i,j] = 1 if surgery i is assigned to block j
        - start[i,j] = start time of surgery i in block j (if assigned)
        
        Constraints:
        - Each surgery assigned to at most one block
        - Surgery durations must fit within block capacity
        - No overlapping surgeries in same block
        - Equipment/specialty compatibility
        
        Objective:
        - Maximize: Σ (priority[i] * x[i,j]) - penalty * idle_time
        """
        import time
        solve_start = time.time()
        
        model = cp_model.CpModel()
        
        num_surgeries = len(surgeries)
        num_blocks = len(blocks)
        
        if num_surgeries == 0 or num_blocks == 0:
            return ScheduleResult(
                status=SchedulingStatus.FEASIBLE,
                scheduled_surgeries=[],
                unscheduled_surgeries=surgeries if num_blocks == 0 else [],
                total_idle_minutes=0,
                utilization_percent=0.0,
                solver_time_seconds=0.0,
                objective_value=0.0
            )
        
        # Buffer time between surgeries
        buffer_time = self.config.surgery_buffer_minutes + self.config.cleanup_time_minutes
        
        # Decision variables
        # x[i][j] = 1 if surgery i is assigned to block j
        x = {}
        for i in range(num_surgeries):
            for j in range(num_blocks):
                x[i, j] = model.NewBoolVar(f'surgery_{i}_block_{j}')
        
        # Start time variables (within block, in minutes from block start)
        start_times = {}
        for i in range(num_surgeries):
            for j in range(num_blocks):
                max_start = max(0, blocks[j].capacity_minutes - surgeries[i].predicted_duration)
                start_times[i, j] = model.NewIntVar(0, max_start, f'start_{i}_{j}')
        
        # Constraint 1: Each surgery assigned to at most one block
        for i in range(num_surgeries):
            model.Add(sum(x[i, j] for j in range(num_blocks)) <= 1)
        
        # Constraint 2: Compatibility constraints
        for i in range(num_surgeries):
            for j in range(num_blocks):
                if not self.check_compatibility(surgeries[i], blocks[j]):
                    model.Add(x[i, j] == 0)
        
        # Constraint 3: Capacity constraints (bin packing)
        for j in range(num_blocks):
            # Total duration of all surgeries in block j must fit
            total_duration = sum(
                x[i, j] * (surgeries[i].predicted_duration + buffer_time)
                for i in range(num_surgeries)
            )
            model.Add(total_duration <= blocks[j].capacity_minutes)
        
        # Constraint 4: No overlapping surgeries within a block
        # Using interval variables for each surgery-block assignment
        intervals = {}
        for i in range(num_surgeries):
            for j in range(num_blocks):
                duration = surgeries[i].predicted_duration + buffer_time
                
                # Create interval only if assigned
                interval = model.NewOptionalIntervalVar(
                    start_times[i, j],
                    duration,
                    start_times[i, j] + duration,
                    x[i, j],
                    f'interval_{i}_{j}'
                )
                intervals[i, j] = interval
        
        # No overlap constraint for each block
        for j in range(num_blocks):
            block_intervals = [intervals[i, j] for i in range(num_surgeries)]
            model.AddNoOverlap(block_intervals)
        
        # Constraint 5: Emergency surgeries must be scheduled
        for i in range(num_surgeries):
            if surgeries[i].is_emergency:
                model.Add(sum(x[i, j] for j in range(num_blocks)) == 1)
        
        # Objective: Maximize weighted sum of scheduled surgeries
        # Weight = priority score (higher acuity = higher weight)
        objective_terms = []
        for i in range(num_surgeries):
            for j in range(num_blocks):
                weight = int(surgeries[i].priority_score * 100)
                objective_terms.append(weight * x[i, j])
        
        # Add penalty for idle time (encourage filling blocks)
        idle_time_vars = []
        for j in range(num_blocks):
            used_time = sum(
                x[i, j] * (surgeries[i].predicted_duration + buffer_time)
                for i in range(num_surgeries)
            )
            idle_var = model.NewIntVar(0, blocks[j].capacity_minutes, f'idle_{j}')
            model.Add(idle_var == blocks[j].capacity_minutes - used_time)
            idle_time_vars.append(idle_var)
        
        # Objective: maximize priority - penalty * idle_time
        idle_penalty = 10  # Weight for idle time penalty
        model.Maximize(
            sum(objective_terms) - idle_penalty * sum(idle_time_vars)
        )
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.config.max_solver_time_seconds
        status = solver.Solve(model)
        
        solve_time = time.time() - solve_start
        
        # Process results
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            scheduled = []
            unscheduled = []
            total_idle = 0
            total_used = 0
            total_capacity = sum(b.capacity_minutes for b in blocks)
            
            for i in range(num_surgeries):
                is_scheduled = False
                for j in range(num_blocks):
                    if solver.Value(x[i, j]) == 1:
                        is_scheduled = True
                        start_offset = solver.Value(start_times[i, j])
                        
                        scheduled_start = blocks[j].start_time + timedelta(minutes=start_offset)
                        scheduled_end = scheduled_start + timedelta(
                            minutes=surgeries[i].predicted_duration
                        )
                        
                        scheduled.append(ScheduledSurgery(
                            surgery=surgeries[i],
                            block_id=blocks[j].block_id,
                            room_id=blocks[j].room_id,
                            scheduled_start=scheduled_start,
                            scheduled_end=scheduled_end,
                            buffer_after=buffer_time
                        ))
                        
                        total_used += surgeries[i].predicted_duration + buffer_time
                        break
                
                if not is_scheduled:
                    unscheduled.append(surgeries[i])
            
            total_idle = total_capacity - total_used
            utilization = (total_used / total_capacity * 100) if total_capacity > 0 else 0
            
            return ScheduleResult(
                status=SchedulingStatus.OPTIMAL if status == cp_model.OPTIMAL else SchedulingStatus.FEASIBLE,
                scheduled_surgeries=scheduled,
                unscheduled_surgeries=unscheduled,
                total_idle_minutes=max(0, total_idle),
                utilization_percent=round(utilization, 2),
                solver_time_seconds=round(solve_time, 3),
                objective_value=solver.ObjectiveValue()
            )
        
        elif status == cp_model.INFEASIBLE:
            return ScheduleResult(
                status=SchedulingStatus.INFEASIBLE,
                scheduled_surgeries=[],
                unscheduled_surgeries=surgeries,
                total_idle_minutes=0,
                utilization_percent=0.0,
                solver_time_seconds=round(solve_time, 3),
                objective_value=0.0
            )
        
        else:
            return ScheduleResult(
                status=SchedulingStatus.TIMEOUT,
                scheduled_surgeries=[],
                unscheduled_surgeries=surgeries,
                total_idle_minutes=0,
                utilization_percent=0.0,
                solver_time_seconds=round(solve_time, 3),
                objective_value=0.0
            )
    
    def schedule(
        self,
        surgeries: List[Surgery],
        start_date: Optional[datetime] = None,
        num_days: int = 1,
        or_blocks: Optional[List[ORBlock]] = None
    ) -> ScheduleResult:
        """
        Main scheduling entry point.
        
        Takes a list of surgeries and returns an optimized schedule.
        """
        start_date = start_date or datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)  # Default to tomorrow
        
        # Prepare surgeries with predicted durations and priorities
        prepared_surgeries = self.prepare_surgeries(surgeries)
        
        # Generate OR blocks if not provided
        blocks = or_blocks or self.generate_or_blocks(start_date, num_days)
        
        logger.info(
            f"Scheduling {len(prepared_surgeries)} surgeries across "
            f"{len(blocks)} OR blocks"
        )
        
        # Solve the bin packing problem
        result = self.solve_bin_packing(prepared_surgeries, blocks)
        
        logger.info(
            f"Scheduling complete: {len(result.scheduled_surgeries)} scheduled, "
            f"{len(result.unscheduled_surgeries)} unscheduled, "
            f"{result.utilization_percent}% utilization"
        )
        
        return result


class ERQueueManager:
    """
    Emergency Room Queue Manager.
    
    Manages the ER waiting queue by reordering patients based on
    triage scores and acuity levels.
    """
    
    def __init__(self):
        """Initialize the queue manager."""
        self.acuity_weights = config.acuity.weights
    
    def calculate_queue_score(
        self,
        patient_acuity: int,
        triage_score: float,
        wait_time_minutes: int,
        is_pediatric: bool = False,
        is_geriatric: bool = False
    ) -> float:
        """
        Calculate queue priority score.
        
        Combines multiple factors:
        - Acuity level (ESI 1-5)
        - Triage agent score
        - Wait time penalty
        - Special population bonuses
        """
        # Base score from acuity
        acuity_name = config.acuity.esi_mapping.get(patient_acuity, "urgent")
        acuity_score = self.acuity_weights.get(acuity_name, 50)
        
        # Triage score contribution (0-100 scale)
        triage_contribution = triage_score * 100
        
        # Wait time bonus (1 point per 5 minutes waited)
        wait_bonus = wait_time_minutes / 5
        
        # Special population bonuses
        special_bonus = 0
        if is_pediatric:
            special_bonus += 10
        if is_geriatric:
            special_bonus += 5
        
        # Combined score (higher = higher priority)
        total_score = (
            acuity_score * 2 +  # Acuity weighted heavily
            triage_contribution +
            wait_bonus +
            special_bonus
        )
        
        return round(total_score, 2)
    
    def reorder_queue(
        self,
        patients: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Reorder ER queue based on priority scores.
        
        Input patients should have:
        - patient_id: str
        - acuity: int (1-5)
        - triage_score: float (0-1)
        - wait_time_minutes: int
        - is_pediatric: bool (optional)
        - is_geriatric: bool (optional)
        
        Returns sorted list with added queue_position and priority_score.
        """
        scored_patients = []
        
        for patient in patients:
            score = self.calculate_queue_score(
                patient_acuity=patient.get("acuity", 3),
                triage_score=patient.get("triage_score", 0.5),
                wait_time_minutes=patient.get("wait_time_minutes", 0),
                is_pediatric=patient.get("is_pediatric", False),
                is_geriatric=patient.get("is_geriatric", False)
            )
            
            scored_patient = {**patient, "priority_score": score}
            scored_patients.append(scored_patient)
        
        # Sort by priority score (descending)
        sorted_patients = sorted(
            scored_patients,
            key=lambda p: p["priority_score"],
            reverse=True
        )
        
        # Add queue positions
        for idx, patient in enumerate(sorted_patients):
            patient["queue_position"] = idx + 1
        
        return sorted_patients
