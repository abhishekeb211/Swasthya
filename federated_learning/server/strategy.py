"""
Federated Learning Server - Custom Aggregation Strategy

This module implements BestModelStrategy, a custom aggregation strategy that
improves upon standard Federated Averaging (FedAvg) for healthcare deployments.

================================================================================
DATA SOVEREIGNTY: Code Travels to Data, Not Data to Code
================================================================================

In traditional centralized machine learning, the data pipeline looks like this:

    Hospital A ──────┐
    (Patient Data)   │
                     │      ┌─────────────────────────┐
    Hospital B ──────┼─────►│   Central ML Server     │
    (Patient Data)   │      │   • Aggregates ALL data │
                     │      │   • Trains ONE model    │
    Hospital C ──────┘      │   • Full data access    │
    (Patient Data)          └─────────────────────────┘
    
    PROBLEMS:
    • HIPAA/GDPR violations: Patient data leaves hospital networks
    • Single point of failure: Central server compromise exposes ALL data
    • Network bottleneck: Terabytes of medical records transferred
    • Consent issues: Patients may consent to treatment, not data sharing

With FEDERATED LEARNING, the paradigm inverts:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         FL Server (This Module)                          │
    │   • Sends MODEL CODE to each hospital                                    │
    │   • Receives only MODEL WEIGHTS (numbers, not data)                      │
    │   • Aggregates weights using BestModelStrategy                           │
    │   • NEVER sees any patient data                                          │
    └───────────────────────────────┬─────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │   Hospital A    │   │   Hospital B    │   │   Hospital C    │
    │                 │   │                 │   │                 │
    │ [Patient Data]  │   │ [Patient Data]  │   │ [Patient Data]  │
    │      │          │   │      │          │   │      │          │
    │      ▼          │   │      ▼          │   │      ▼          │
    │ [Local Model]   │   │ [Local Model]   │   │ [Local Model]   │
    │      │          │   │      │          │   │      │          │
    │      ▼          │   │      ▼          │   │      ▼          │
    │ Send Weights ───┼───┼──►  Weights  ◄──┼───┼─── Weights      │
    │ (Not Data!)     │   │                 │   │                 │
    └─────────────────┘   └─────────────────┘   └─────────────────┘
    
    BENEFITS:
    ✓ Patient data NEVER leaves the hospital
    ✓ Compliance with HIPAA, GDPR, CCPA by design
    ✓ Each hospital maintains data sovereignty
    ✓ Collective intelligence without data pooling
    ✓ Works even if hospitals are competitors

================================================================================
WHY STANDARD FedAvg CAN FAIL IN HEALTHCARE
================================================================================

Standard Federated Averaging computes the global model as:

    w_global = Σ (n_k / n_total) * w_k
    
    where:
    - w_k = model weights from hospital k
    - n_k = number of training samples at hospital k
    - n_total = total samples across all hospitals

PROBLEM 1: Data Quality Variance
─────────────────────────────────
Not all hospital data is created equal:

    Hospital A (Academic Medical Center):
    ├── n = 100,000 patients
    ├── High-quality EHR with structured data
    ├── Rigorous data governance
    └── Diverse patient population

    Hospital B (Rural Community Hospital):
    ├── n = 5,000 patients
    ├── Older EHR system with missing fields
    ├── Less standardized documentation
    └── Homogeneous patient population

In standard FedAvg, Hospital A dominates (100K / 105K = 95% weight).
But what if Hospital B has better data QUALITY despite less QUANTITY?

PROBLEM 2: Distribution Shift
─────────────────────────────
Different hospitals see different patient populations:

    Pediatric Hospital:   90% patients < 18 years old
    Veterans Hospital:    80% patients > 50 years old
    Urban Trauma Center:  40% gunshot/stab wounds
    Suburban Hospital:    60% elective procedures

A model weighted purely by sample size may perform poorly
on minority populations at smaller specialty hospitals.

PROBLEM 3: Adversarial/Corrupted Clients
────────────────────────────────────────
What if one hospital has:
- Compromised systems sending malicious weights
- Systematic data entry errors
- Outdated data from a different era

Standard FedAvg blindly incorporates these updates.

================================================================================
SOLUTION: BestModelStrategy - Performance-Weighted Aggregation
================================================================================

Our BestModelStrategy modifies aggregation to consider model PERFORMANCE:

    w_global = Σ (performance_score_k / total_score) * w_k
    
    where performance_score_k considers:
    - Validation metrics (MAE, RMSE, accuracy)
    - Sample diversity indicators
    - Historical reliability of this client
    - Anomaly detection on weight updates

This ensures:
1. Hospitals with better-performing models contribute more
2. Poor quality data sources are automatically down-weighted
3. The global model improves even if some clients have issues
4. Smaller hospitals with unique data can still contribute meaningfully

================================================================================
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ClientPerformanceTracker:
    """
    Tracks historical performance of FL clients across training rounds.
    
    This enables intelligent weighting decisions based on:
    - Rolling average of client evaluation metrics
    - Consistency of client contributions
    - Anomaly detection for potentially corrupted updates
    
    Attributes:
        history: Dict mapping client_id to list of (round, metrics) tuples
        reliability_scores: Dict mapping client_id to reliability score [0, 1]
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the performance tracker.
        
        Args:
            window_size: Number of recent rounds to consider for rolling metrics
        """
        self.window_size = window_size
        self.history: Dict[str, List[Tuple[int, Dict[str, float]]]] = {}
        self.reliability_scores: Dict[str, float] = {}
        self.round_count = 0
    
    def record_client_metrics(
        self,
        client_id: str,
        round_num: int,
        metrics: Dict[str, float],
    ) -> None:
        """Record metrics from a client's training round."""
        if client_id not in self.history:
            self.history[client_id] = []
            self.reliability_scores[client_id] = 1.0  # Start with full trust
        
        self.history[client_id].append((round_num, metrics.copy()))
        
        # Keep only recent history
        if len(self.history[client_id]) > self.window_size * 2:
            self.history[client_id] = self.history[client_id][-self.window_size * 2:]
        
        # Update reliability based on metric consistency
        self._update_reliability(client_id)
    
    def _update_reliability(self, client_id: str) -> None:
        """
        Update reliability score based on metric consistency.
        
        Clients with erratic or degrading metrics get lower reliability.
        """
        history = self.history.get(client_id, [])
        if len(history) < 2:
            return
        
        # Get recent metrics
        recent = history[-self.window_size:]
        
        # Calculate consistency (lower variance = higher reliability)
        mae_values = [m.get("mae", 0) for _, m in recent if "mae" in m]
        if mae_values and len(mae_values) >= 2:
            variance = np.var(mae_values)
            mean_mae = np.mean(mae_values)
            
            # Coefficient of variation (normalized variance)
            cv = np.sqrt(variance) / (mean_mae + 1e-10)
            
            # Convert to reliability score (lower cv = higher reliability)
            # cv < 0.1 → reliability ≈ 1.0
            # cv > 1.0 → reliability ≈ 0.5
            reliability = max(0.5, 1.0 - cv * 0.5)
            
            # Smooth update (exponential moving average)
            alpha = 0.3
            self.reliability_scores[client_id] = (
                alpha * reliability + 
                (1 - alpha) * self.reliability_scores.get(client_id, 1.0)
            )
    
    def get_client_weight_multiplier(self, client_id: str) -> float:
        """
        Get weight multiplier for a client based on historical performance.
        
        Returns:
            Float multiplier in range [0.5, 1.5]
            - 0.5: Historically poor/unreliable client
            - 1.0: Average client
            - 1.5: Historically excellent client
        """
        reliability = self.reliability_scores.get(client_id, 1.0)
        
        # Convert reliability [0.5, 1.0] to multiplier [0.5, 1.5]
        multiplier = reliability * 1.5
        
        return max(0.5, min(1.5, multiplier))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of tracked client performance."""
        return {
            "total_clients_tracked": len(self.history),
            "reliability_scores": dict(self.reliability_scores),
            "rounds_tracked": self.round_count,
        }


class BestModelStrategy(FedAvg):
    """
    Custom Federated Averaging strategy that prioritizes best-performing models.
    
    This strategy extends FedAvg with several improvements:
    
    1. PERFORMANCE-WEIGHTED AGGREGATION:
       Instead of weighting purely by sample count, we incorporate
       evaluation metrics (MAE, RMSE) into the weighting scheme.
    
    2. CLIENT RELIABILITY TRACKING:
       Clients with historically consistent, high-quality updates
       receive higher weights in aggregation.
    
    3. OUTLIER DETECTION:
       Anomalous weight updates (potentially from corrupted clients)
       are detected and down-weighted.
    
    4. ADAPTIVE LEARNING:
       The strategy adapts its aggregation approach based on
       observed client behavior over training rounds.
    
    Inherits from FedAvg to maintain compatibility with standard
    Flower workflows while adding healthcare-specific optimizations.
    
    Example:
        >>> strategy = BestModelStrategy(
        ...     fraction_fit=0.8,
        ...     min_fit_clients=2,
        ...     performance_weight=0.3,
        ... )
        >>> fl.server.start_server(strategy=strategy, ...)
    
    Attributes:
        performance_weight: How much to weight performance vs sample count (0-1)
        performance_tracker: Tracks client metrics over time
        best_global_metrics: Best metrics achieved by global model
        anomaly_threshold: Z-score threshold for detecting anomalous updates
    """
    
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        # Custom parameters for BestModelStrategy
        performance_weight: float = 0.3,
        use_reliability_tracking: bool = True,
        anomaly_threshold: float = 3.0,
        metric_for_ranking: str = "mae",
        lower_is_better: bool = True,
    ) -> None:
        """
        Initialize BestModelStrategy with custom aggregation parameters.
        
        Args:
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum number of clients for training round
            min_evaluate_clients: Minimum number of clients for evaluation
            min_available_clients: Minimum available clients to start round
            evaluate_fn: Server-side model evaluation function
            on_fit_config_fn: Function to configure client fit
            on_evaluate_config_fn: Function to configure client evaluation
            accept_failures: Whether to continue if some clients fail
            initial_parameters: Initial global model parameters
            fit_metrics_aggregation_fn: Function to aggregate fit metrics
            evaluate_metrics_aggregation_fn: Function to aggregate eval metrics
            
            # BestModelStrategy-specific:
            performance_weight: Weight for performance-based aggregation [0, 1]
                0.0 = pure sample-weighted (like standard FedAvg)
                1.0 = pure performance-weighted
                0.3 = recommended balance (default)
            use_reliability_tracking: Whether to track client reliability
            anomaly_threshold: Z-score threshold for anomaly detection
            metric_for_ranking: Which metric to use for ranking ("mae", "rmse", etc.)
            lower_is_better: Whether lower metric values indicate better performance
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        
        # Custom strategy parameters
        self.performance_weight = performance_weight
        self.use_reliability_tracking = use_reliability_tracking
        self.anomaly_threshold = anomaly_threshold
        self.metric_for_ranking = metric_for_ranking
        self.lower_is_better = lower_is_better
        
        # Internal state
        self.performance_tracker = ClientPerformanceTracker()
        self.best_global_metrics: Optional[Dict[str, float]] = None
        self.current_round = 0
        self.aggregation_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"BestModelStrategy initialized: "
            f"performance_weight={performance_weight}, "
            f"reliability_tracking={use_reliability_tracking}, "
            f"metric={metric_for_ranking} ({'lower' if lower_is_better else 'higher'} is better)"
        )
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model updates from clients using performance-weighted averaging.
        
        This method overrides FedAvg.aggregate_fit to implement:
        1. Performance-based weighting
        2. Reliability score incorporation
        3. Anomaly detection and mitigation
        
        Args:
            server_round: Current round number
            results: List of (client_proxy, fit_result) tuples for successful clients
            failures: List of failed client interactions
        
        Returns:
            Tuple of (aggregated_parameters, aggregated_metrics)
        """
        self.current_round = server_round
        
        if not results:
            logger.warning(f"Round {server_round}: No results to aggregate")
            return None, {}
        
        # Log round information
        logger.info(
            f"Round {server_round}: Aggregating {len(results)} client updates "
            f"({len(failures)} failures)"
        )
        
        # Extract client information
        client_data = []
        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics or {}
            parameters = fit_res.parameters
            
            # Record metrics for reliability tracking
            if self.use_reliability_tracking and metrics:
                self.performance_tracker.record_client_metrics(
                    client_id, server_round, {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
                )
            
            client_data.append({
                "client_id": client_id,
                "num_examples": num_examples,
                "metrics": metrics,
                "parameters": parameters,
                "ndarrays": parameters_to_ndarrays(parameters),
            })
        
        # Calculate weights using BestModel strategy
        weights = self._calculate_aggregation_weights(client_data)
        
        # Check for anomalous updates
        if len(client_data) >= 3:
            weights = self._apply_anomaly_detection(client_data, weights)
        
        # Aggregate parameters
        aggregated_ndarrays = self._weighted_average(
            [cd["ndarrays"] for cd in client_data],
            weights,
        )
        
        # Convert back to Parameters
        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        
        # Aggregate metrics (weighted average)
        aggregated_metrics = self._aggregate_metrics(client_data, weights)
        
        # Record aggregation for analysis
        self._record_aggregation(server_round, client_data, weights, aggregated_metrics)
        
        logger.info(
            f"Round {server_round} aggregation complete. "
            f"Weights: {dict(zip([cd['client_id'] for cd in client_data], [f'{w:.3f}' for w in weights]))}"
        )
        
        return aggregated_parameters, aggregated_metrics
    
    def _calculate_aggregation_weights(
        self,
        client_data: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Calculate aggregation weights combining sample count and performance.
        
        The weight for client k is:
        
            w_k = α * perf_weight_k + (1-α) * sample_weight_k
            
        where:
            - α = self.performance_weight
            - perf_weight_k = normalized performance score
            - sample_weight_k = n_k / Σn_i (standard FedAvg weight)
        
        Returns:
            List of normalized weights summing to 1.0
        """
        num_clients = len(client_data)
        
        # Calculate sample-based weights (standard FedAvg)
        total_examples = sum(cd["num_examples"] for cd in client_data)
        sample_weights = [
            cd["num_examples"] / total_examples if total_examples > 0 else 1 / num_clients
            for cd in client_data
        ]
        
        # Calculate performance-based weights
        performance_weights = self._calculate_performance_weights(client_data)
        
        # Apply reliability multipliers
        if self.use_reliability_tracking:
            reliability_multipliers = [
                self.performance_tracker.get_client_weight_multiplier(cd["client_id"])
                for cd in client_data
            ]
        else:
            reliability_multipliers = [1.0] * num_clients
        
        # Combine weights
        α = self.performance_weight
        combined_weights = [
            (α * pw + (1 - α) * sw) * rm
            for pw, sw, rm in zip(performance_weights, sample_weights, reliability_multipliers)
        ]
        
        # Normalize to sum to 1
        total_weight = sum(combined_weights)
        normalized_weights = [w / total_weight for w in combined_weights]
        
        return normalized_weights
    
    def _calculate_performance_weights(
        self,
        client_data: List[Dict[str, Any]],
    ) -> List[float]:
        """
        Calculate weights based on client model performance metrics.
        
        Uses the configured metric (default: MAE) to rank clients.
        Better-performing clients receive higher weights.
        """
        num_clients = len(client_data)
        
        # Extract performance metric values
        metric_values = []
        for cd in client_data:
            metrics = cd.get("metrics", {})
            value = metrics.get(self.metric_for_ranking)
            
            if value is not None and isinstance(value, (int, float)):
                metric_values.append(float(value))
            else:
                # Use None to indicate missing metric
                metric_values.append(None)
        
        # If no valid metrics, return uniform weights
        valid_values = [v for v in metric_values if v is not None]
        if not valid_values:
            return [1 / num_clients] * num_clients
        
        # Calculate performance scores
        # Convert metrics to scores where higher is better
        if self.lower_is_better:
            # For metrics like MAE/RMSE, invert so lower values get higher scores
            max_value = max(valid_values) + 1e-10
            scores = [
                (max_value - v) / max_value if v is not None else 0.5
                for v in metric_values
            ]
        else:
            # For metrics like accuracy, use directly
            min_value = min(valid_values) - 1e-10
            max_value = max(valid_values) + 1e-10
            scores = [
                (v - min_value) / (max_value - min_value) if v is not None else 0.5
                for v in metric_values
            ]
        
        # Apply softmax for smooth weighting
        # Temperature controls how much we favor top performers
        temperature = 0.5
        exp_scores = [np.exp(s / temperature) for s in scores]
        total_exp = sum(exp_scores)
        
        weights = [e / total_exp for e in exp_scores]
        
        return weights
    
    def _apply_anomaly_detection(
        self,
        client_data: List[Dict[str, Any]],
        weights: List[float],
    ) -> List[float]:
        """
        Detect and down-weight anomalous client updates.
        
        Uses statistical analysis to identify clients whose weight updates
        significantly deviate from the norm. Such updates might indicate:
        - Corrupted data at the client
        - Malicious clients (adversarial attacks)
        - Significantly different data distributions
        
        Returns:
            Adjusted weights with anomalies down-weighted
        """
        num_clients = len(client_data)
        
        # Calculate weight norms for each client
        weight_norms = []
        for cd in client_data:
            ndarrays = cd["ndarrays"]
            if ndarrays:
                norm = sum(np.linalg.norm(arr) for arr in ndarrays)
                weight_norms.append(norm)
            else:
                weight_norms.append(0)
        
        # Calculate z-scores
        if len(weight_norms) >= 3:
            mean_norm = np.mean(weight_norms)
            std_norm = np.std(weight_norms)
            
            if std_norm > 1e-10:
                z_scores = [(n - mean_norm) / std_norm for n in weight_norms]
            else:
                z_scores = [0] * num_clients
        else:
            z_scores = [0] * num_clients
        
        # Adjust weights for anomalies
        adjusted_weights = []
        for i, (weight, z_score) in enumerate(zip(weights, z_scores)):
            if abs(z_score) > self.anomaly_threshold:
                # Down-weight anomalous updates
                adjustment = 0.1  # Reduce to 10% of original weight
                adjusted_weight = weight * adjustment
                
                client_id = client_data[i]["client_id"]
                logger.warning(
                    f"Anomaly detected for client {client_id}: "
                    f"z-score={z_score:.2f}, weight adjusted {weight:.3f} -> {adjusted_weight:.3f}"
                )
            else:
                adjusted_weight = weight
            
            adjusted_weights.append(adjusted_weight)
        
        # Re-normalize
        total = sum(adjusted_weights)
        if total > 0:
            adjusted_weights = [w / total for w in adjusted_weights]
        
        return adjusted_weights
    
    def _weighted_average(
        self,
        ndarrays_list: List[NDArrays],
        weights: List[float],
    ) -> NDArrays:
        """
        Compute weighted average of model parameters.
        
        This is the core aggregation operation that produces the new global model.
        """
        if not ndarrays_list:
            return []
        
        # Initialize with zeros
        num_layers = len(ndarrays_list[0])
        averaged = [
            np.zeros_like(ndarrays_list[0][i])
            for i in range(num_layers)
        ]
        
        # Weighted sum
        for ndarrays, weight in zip(ndarrays_list, weights):
            for i in range(num_layers):
                averaged[i] += weight * ndarrays[i]
        
        return averaged
    
    def _aggregate_metrics(
        self,
        client_data: List[Dict[str, Any]],
        weights: List[float],
    ) -> Dict[str, Scalar]:
        """Aggregate client metrics using weights."""
        aggregated = {}
        
        # Collect all metric keys
        all_keys = set()
        for cd in client_data:
            all_keys.update(cd.get("metrics", {}).keys())
        
        for key in all_keys:
            values_weights = []
            for cd, w in zip(client_data, weights):
                value = cd.get("metrics", {}).get(key)
                if isinstance(value, (int, float)):
                    values_weights.append((float(value), w))
            
            if values_weights:
                total_weight = sum(w for _, w in values_weights)
                if total_weight > 0:
                    aggregated[key] = sum(v * w for v, w in values_weights) / total_weight
        
        # Add strategy metadata
        aggregated["strategy"] = "BestModelStrategy"
        aggregated["round"] = self.current_round
        aggregated["num_clients"] = len(client_data)
        
        return aggregated
    
    def _record_aggregation(
        self,
        server_round: int,
        client_data: List[Dict[str, Any]],
        weights: List[float],
        aggregated_metrics: Dict[str, Scalar],
    ) -> None:
        """Record aggregation details for analysis and debugging."""
        record = {
            "round": server_round,
            "timestamp": datetime.utcnow().isoformat(),
            "num_clients": len(client_data),
            "client_weights": {
                cd["client_id"]: {
                    "weight": w,
                    "num_examples": cd["num_examples"],
                    "metrics": cd.get("metrics", {}),
                }
                for cd, w in zip(client_data, weights)
            },
            "aggregated_metrics": dict(aggregated_metrics),
        }
        
        self.aggregation_history.append(record)
        
        # Keep only recent history
        if len(self.aggregation_history) > 100:
            self.aggregation_history = self.aggregation_history[-100:]
    
    def get_aggregation_summary(self) -> Dict[str, Any]:
        """Get summary of aggregation history for monitoring."""
        return {
            "total_rounds": len(self.aggregation_history),
            "recent_rounds": self.aggregation_history[-5:] if self.aggregation_history else [],
            "performance_tracker": self.performance_tracker.get_summary(),
            "strategy_config": {
                "performance_weight": self.performance_weight,
                "reliability_tracking": self.use_reliability_tracking,
                "anomaly_threshold": self.anomaly_threshold,
                "metric_for_ranking": self.metric_for_ranking,
                "lower_is_better": self.lower_is_better,
            },
        }


def create_demand_strategy(
    min_clients: int = 2,
    num_rounds: int = 5,
) -> BestModelStrategy:
    """
    Factory function to create a BestModelStrategy configured for demand forecasting.
    
    Args:
        min_clients: Minimum number of clients required per round
        num_rounds: Total number of training rounds (for config)
    
    Returns:
        Configured BestModelStrategy instance
    """
    return BestModelStrategy(
        fraction_fit=1.0,  # Use all available clients for training
        fraction_evaluate=1.0,  # Evaluate on all clients
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        performance_weight=0.3,
        use_reliability_tracking=True,
        metric_for_ranking="mae",
        lower_is_better=True,
    )


def create_triage_strategy(
    min_clients: int = 2,
    num_rounds: int = 5,
) -> BestModelStrategy:
    """
    Factory function to create a BestModelStrategy configured for triage prediction.
    
    XGBoost triage models are evaluated on accuracy, so we configure accordingly.
    
    Args:
        min_clients: Minimum number of clients required per round
        num_rounds: Total number of training rounds (for config)
    
    Returns:
        Configured BestModelStrategy instance
    """
    return BestModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        performance_weight=0.4,  # Weight performance more for triage
        use_reliability_tracking=True,
        metric_for_ranking="accuracy",
        lower_is_better=False,  # Higher accuracy is better
        anomaly_threshold=2.5,  # Slightly stricter for triage
    )
