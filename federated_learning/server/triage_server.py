"""
Federated Learning Server for Triage Prediction

This module implements the FL server that coordinates triage/discharge
readiness model training across multiple hospital clients.

================================================================================
FL TRIAGE SERVER
================================================================================

Port: 8086 (default)
Protocol: gRPC
Strategy: BestModelStrategy (performance-weighted FedAvg)

The triage server coordinates training of XGBoost-based models that predict:
- Patient acuity levels (ESI 1-5)
- Discharge readiness scores
- Expected length of stay

Training Configuration:
- Rounds: 5 (configurable)
- Min Clients: 2 (configurable)
- Performance Metric: Accuracy (higher is better)

================================================================================
TRIAGE VS DEMAND: WHY SEPARATE SERVERS?
================================================================================

While both servers use federated learning, they handle different model types:

DEMAND SERVER (Port 8087):
├── Model Type: Prophet (time-series)
├── Output: Continuous forecast (patient counts)
├── Metric: MAE/RMSE (lower is better)
├── Parameters: Trend + seasonality coefficients
└── Use Case: Capacity planning

TRIAGE SERVER (Port 8086):
├── Model Type: XGBoost (gradient boosting)
├── Output: Classification (acuity levels)
├── Metric: Accuracy/AUC (higher is better)
├── Parameters: Tree structure + weights
└── Use Case: Patient prioritization

Separating these allows:
1. Independent scaling of training resources
2. Different aggregation strategies per model type
3. Isolated failure domains
4. Model-specific hyperparameter tuning

================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import Metrics, Scalar
from flwr.server import ServerConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from strategy import BestModelStrategy, create_triage_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/fl_triage_server.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class TriageServerConfig:
    """Configuration for the Triage FL Server."""
    
    # Network
    SERVER_ADDRESS: str = os.getenv("FL_TRIAGE_SERVER_ADDRESS", "0.0.0.0:8086")
    GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # 512MB
    
    # Training rounds
    NUM_ROUNDS: int = int(os.getenv("FL_NUM_ROUNDS", "5"))
    MIN_FIT_CLIENTS: int = int(os.getenv("FL_MIN_FIT_CLIENTS", "2"))
    MIN_EVALUATE_CLIENTS: int = int(os.getenv("FL_MIN_EVAL_CLIENTS", "2"))
    MIN_AVAILABLE_CLIENTS: int = int(os.getenv("FL_MIN_AVAILABLE_CLIENTS", "2"))
    
    # Timeouts
    CLIENT_TIMEOUT: int = int(os.getenv("FL_CLIENT_TIMEOUT", "300"))
    ROUND_TIMEOUT: int = int(os.getenv("FL_ROUND_TIMEOUT", "600"))
    
    # Model persistence
    MODEL_SAVE_PATH: str = os.getenv("FL_MODEL_SAVE_PATH", "/tmp/fl_triage_model")


# =============================================================================
# METRICS AGGREGATION
# =============================================================================

def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluation metrics from multiple clients.
    
    For triage models, we track:
    - accuracy: Overall classification accuracy
    - auc: Area under ROC curve
    - precision: Weighted precision
    - recall: Weighted recall
    - f1: F1 score
    """
    if not metrics:
        return {}
    
    total_examples = sum(num for num, _ in metrics)
    
    if total_examples == 0:
        return {"error": "No examples received"}
    
    # Collect all metric keys
    all_keys = set()
    for _, m in metrics:
        all_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
    
    # Calculate weighted averages
    aggregated = {}
    for key in all_keys:
        weighted_sum = 0.0
        weight_sum = 0
        
        for num_examples, client_metrics in metrics:
            if key in client_metrics:
                value = client_metrics[key]
                if isinstance(value, (int, float)):
                    weighted_sum += num_examples * float(value)
                    weight_sum += num_examples
        
        if weight_sum > 0:
            aggregated[key] = weighted_sum / weight_sum
    
    # Add metadata
    aggregated["total_examples"] = total_examples
    aggregated["num_clients"] = len(metrics)
    aggregated["aggregation_time"] = datetime.utcnow().isoformat()
    
    # Log key metrics
    if "accuracy" in aggregated:
        logger.info(f"Aggregated accuracy: {aggregated['accuracy']:.4f}")
    if "auc" in aggregated:
        logger.info(f"Aggregated AUC: {aggregated['auc']:.4f}")
    
    return aggregated


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics with triage-specific logging."""
    aggregated = weighted_average_metrics(metrics)
    aggregated["fit_phase"] = True
    
    # Track training loss if available
    if "loss" in aggregated:
        logger.info(f"Training round - Aggregated loss: {aggregated['loss']:.4f}")
    
    return aggregated


# =============================================================================
# SERVER IMPLEMENTATION
# =============================================================================

def create_server_config() -> ServerConfig:
    """Create Flower server configuration."""
    return ServerConfig(
        num_rounds=TriageServerConfig.NUM_ROUNDS,
        round_timeout=TriageServerConfig.ROUND_TIMEOUT,
    )


def create_strategy() -> BestModelStrategy:
    """
    Create BestModelStrategy configured for triage/classification models.
    
    Key differences from demand strategy:
    - metric_for_ranking: "accuracy" (higher is better)
    - performance_weight: 0.4 (weight performance more heavily)
    - anomaly_threshold: 2.5 (stricter filtering)
    """
    strategy = create_triage_strategy(
        min_clients=TriageServerConfig.MIN_FIT_CLIENTS,
        num_rounds=TriageServerConfig.NUM_ROUNDS,
    )
    
    # Configure metrics aggregation
    strategy.evaluate_metrics_aggregation_fn = weighted_average_metrics
    strategy.fit_metrics_aggregation_fn = fit_metrics_aggregation
    
    logger.info(
        f"Created Triage Strategy: "
        f"min_clients={TriageServerConfig.MIN_FIT_CLIENTS}, "
        f"rounds={TriageServerConfig.NUM_ROUNDS}, "
        f"metric=accuracy (higher is better)"
    )
    
    return strategy


def save_global_model(
    parameters: Optional[Any],
    round_num: int,
    metrics: Dict[str, Scalar],
) -> None:
    """Save the global XGBoost model after aggregation."""
    if parameters is None:
        logger.warning("No parameters to save")
        return
    
    save_dir = Path(TriageServerConfig.MODEL_SAVE_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / f"global_triage_model_round_{round_num}.npz"
    
    try:
        from flwr.common import parameters_to_ndarrays
        
        ndarrays = parameters_to_ndarrays(parameters)
        
        np.savez_compressed(
            model_path,
            *ndarrays,
            round=round_num,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        logger.info(f"Saved global triage model to {model_path}")
        
        # Save metrics
        metrics_path = save_dir / f"triage_metrics_round_{round_num}.txt"
        with open(metrics_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
                
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def start_triage_server(
    server_address: Optional[str] = None,
    num_rounds: Optional[int] = None,
    min_clients: Optional[int] = None,
) -> None:
    """
    Start the Federated Learning server for triage prediction.
    
    This server coordinates XGBoost model training across hospital clients
    for patient triage and discharge readiness prediction.
    
    Args:
        server_address: Address to bind (e.g., "0.0.0.0:8086")
        num_rounds: Number of training rounds
        min_clients: Minimum number of clients required
    """
    # Override config if parameters provided
    if server_address:
        TriageServerConfig.SERVER_ADDRESS = server_address
    if num_rounds:
        TriageServerConfig.NUM_ROUNDS = num_rounds
    if min_clients:
        TriageServerConfig.MIN_FIT_CLIENTS = min_clients
        TriageServerConfig.MIN_EVALUATE_CLIENTS = min_clients
        TriageServerConfig.MIN_AVAILABLE_CLIENTS = min_clients
    
    # Log startup banner
    print_startup_banner()
    
    # Create strategy and config
    strategy = create_strategy()
    config = create_server_config()
    
    logger.info(f"Starting FL Triage Server on {TriageServerConfig.SERVER_ADDRESS}")
    
    try:
        fl.server.start_server(
            server_address=TriageServerConfig.SERVER_ADDRESS,
            config=config,
            strategy=strategy,
            grpc_max_message_length=TriageServerConfig.GRPC_MAX_MESSAGE_LENGTH,
        )
        
        logger.info("FL Triage Server completed successfully")
        
        # Get final summary
        summary = strategy.get_aggregation_summary()
        logger.info(f"Final aggregation summary: {summary}")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


def print_startup_banner() -> None:
    """Print startup banner."""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║             FEDERATED LEARNING SERVER - TRIAGE PREDICTION                    ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   Server Address:    {TriageServerConfig.SERVER_ADDRESS:<52} ║
    ║   Training Rounds:   {TriageServerConfig.NUM_ROUNDS:<52} ║
    ║   Min Clients:       {TriageServerConfig.MIN_FIT_CLIENTS:<52} ║
    ║   Strategy:          BestModelStrategy (Accuracy-Optimized)                 ║
    ║   Model Type:        XGBoost Classifier                                      ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   TRIAGE MODEL FEATURES:                                                     ║
    ║   • Patient acuity prediction (ESI 1-5)                                      ║
    ║   • Discharge readiness scoring                                              ║
    ║   • Length of stay estimation                                                ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   DATA SOVEREIGNTY GUARANTEE:                                                ║
    ║   • Patient data NEVER leaves hospital networks                              ║
    ║   • Only model parameters (tree weights) are transmitted                     ║
    ║   • HIPAA/GDPR compliant by design                                           ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   WAITING FOR CLIENTS TO CONNECT...                                          ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    Started at: {datetime.utcnow().isoformat()}
    """
    print(banner)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point for the triage FL server."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Server for Patient Triage Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with defaults
  python triage_server.py
  
  # Start with custom configuration
  python triage_server.py --rounds 10 --min-clients 3
  
  # Start on custom address
  python triage_server.py --address 0.0.0.0:9001

Environment Variables:
  FL_TRIAGE_SERVER_ADDRESS  Server bind address (default: 0.0.0.0:8086)
  FL_NUM_ROUNDS            Number of training rounds (default: 5)
  FL_MIN_FIT_CLIENTS       Minimum clients for training (default: 2)
  FL_MODEL_SAVE_PATH       Path to save global models (default: /tmp/fl_triage_model)
        """,
    )
    
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help=f"Server address (default: {TriageServerConfig.SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=f"Number of training rounds (default: {TriageServerConfig.NUM_ROUNDS})",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=None,
        help=f"Minimum number of clients (default: {TriageServerConfig.MIN_FIT_CLIENTS})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    start_triage_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
    )


if __name__ == "__main__":
    main()
