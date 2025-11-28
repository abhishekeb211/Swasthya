"""
Federated Learning Server for Demand Forecasting

This module implements the FL server that coordinates demand forecasting
model training across multiple hospital clients.

================================================================================
FL DEMAND FORECASTING SERVER
================================================================================

Port: 8087 (default)
Protocol: gRPC
Strategy: BestModelStrategy (performance-weighted FedAvg)

Training Configuration:
- Rounds: 5 (configurable)
- Min Clients: 2 (configurable)
- Client Timeout: 300s
- Round Timeout: 600s

The server coordinates the federated learning process:

    Round 1: Initialize
    ├── Wait for min_clients to connect
    ├── Distribute initial model parameters
    └── Each client trains on local data
    
    Round 2-5: Federated Training
    ├── Collect model updates from clients
    ├── Aggregate using BestModelStrategy
    │   ├── Performance-weighted averaging
    │   ├── Reliability tracking
    │   └── Anomaly detection
    ├── Distribute updated global model
    └── Evaluate on each client's test data

    Final: Model Export
    └── Save best global model for deployment

================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common import Metrics, Scalar
from flwr.server import ServerConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from strategy import BestModelStrategy, create_demand_strategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/tmp/fl_demand_server.log", mode="a"),
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

class DemandServerConfig:
    """Configuration for the Demand FL Server."""
    
    # Network
    SERVER_ADDRESS: str = os.getenv("FL_SERVER_ADDRESS", "0.0.0.0:8087")
    GRPC_MAX_MESSAGE_LENGTH: int = 536_870_912  # 512MB for large models
    
    # Training rounds
    NUM_ROUNDS: int = int(os.getenv("FL_NUM_ROUNDS", "5"))
    MIN_FIT_CLIENTS: int = int(os.getenv("FL_MIN_FIT_CLIENTS", "2"))
    MIN_EVALUATE_CLIENTS: int = int(os.getenv("FL_MIN_EVAL_CLIENTS", "2"))
    MIN_AVAILABLE_CLIENTS: int = int(os.getenv("FL_MIN_AVAILABLE_CLIENTS", "2"))
    
    # Timeouts
    CLIENT_TIMEOUT: int = int(os.getenv("FL_CLIENT_TIMEOUT", "300"))  # 5 minutes
    ROUND_TIMEOUT: int = int(os.getenv("FL_ROUND_TIMEOUT", "600"))  # 10 minutes
    
    # Model persistence
    MODEL_SAVE_PATH: str = os.getenv("FL_MODEL_SAVE_PATH", "/tmp/fl_demand_model")
    SAVE_ROUNDS: List[int] = [5]  # Save model after these rounds


# =============================================================================
# METRICS AGGREGATION
# =============================================================================

def weighted_average_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate evaluation metrics from multiple clients using weighted average.
    
    This function is called by the FL server after each evaluation round
    to combine metrics from all participating clients.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples from clients
    
    Returns:
        Dictionary of aggregated metrics
    
    Example:
        >>> metrics = [
        ...     (1000, {"mae": 5.2, "rmse": 7.1}),  # Hospital A
        ...     (500, {"mae": 6.1, "rmse": 8.3}),   # Hospital B
        ... ]
        >>> result = weighted_average_metrics(metrics)
        >>> print(result)
        {"mae": 5.5, "rmse": 7.5, "total_examples": 1500}
    """
    if not metrics:
        return {}
    
    # Extract total examples
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
    
    return aggregated


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate training metrics from multiple clients.
    
    Similar to weighted_average_metrics but for fit (training) phase.
    Also tracks per-round training statistics.
    """
    aggregated = weighted_average_metrics(metrics)
    
    # Add training-specific metadata
    aggregated["fit_phase"] = True
    
    # Log training progress
    if "mae" in aggregated:
        logger.info(f"Training round complete - Aggregated MAE: {aggregated['mae']:.4f}")
    
    return aggregated


# =============================================================================
# SERVER STARTUP
# =============================================================================

def create_server_config() -> ServerConfig:
    """Create Flower server configuration."""
    return ServerConfig(
        num_rounds=DemandServerConfig.NUM_ROUNDS,
        round_timeout=DemandServerConfig.ROUND_TIMEOUT,
    )


def create_strategy() -> BestModelStrategy:
    """
    Create and configure the BestModelStrategy for demand forecasting.
    
    Returns:
        Configured strategy instance with healthcare-optimized parameters
    """
    strategy = create_demand_strategy(
        min_clients=DemandServerConfig.MIN_FIT_CLIENTS,
        num_rounds=DemandServerConfig.NUM_ROUNDS,
    )
    
    # Override with evaluation metrics aggregation
    strategy.evaluate_metrics_aggregation_fn = weighted_average_metrics
    strategy.fit_metrics_aggregation_fn = fit_metrics_aggregation
    
    logger.info(
        f"Created BestModelStrategy: "
        f"min_clients={DemandServerConfig.MIN_FIT_CLIENTS}, "
        f"rounds={DemandServerConfig.NUM_ROUNDS}"
    )
    
    return strategy


def save_global_model(
    parameters: Optional[Any],
    round_num: int,
    metrics: Dict[str, Scalar],
) -> None:
    """
    Save the global model after aggregation.
    
    Args:
        parameters: Aggregated model parameters
        round_num: Current round number
        metrics: Aggregated metrics from this round
    """
    if parameters is None:
        logger.warning("No parameters to save")
        return
    
    save_dir = Path(DemandServerConfig.MODEL_SAVE_PATH)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    model_path = save_dir / f"global_model_round_{round_num}.npz"
    
    try:
        # Convert parameters to numpy arrays if needed
        from flwr.common import parameters_to_ndarrays
        
        ndarrays = parameters_to_ndarrays(parameters)
        
        # Save as compressed numpy archive
        np.savez_compressed(
            model_path,
            *ndarrays,
            round=round_num,
            timestamp=datetime.utcnow().isoformat(),
        )
        
        logger.info(f"Saved global model to {model_path}")
        
        # Save metrics
        metrics_path = save_dir / f"metrics_round_{round_num}.txt"
        with open(metrics_path, "w") as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")


def start_demand_server(
    server_address: Optional[str] = None,
    num_rounds: Optional[int] = None,
    min_clients: Optional[int] = None,
) -> None:
    """
    Start the Federated Learning server for demand forecasting.
    
    This function initializes and runs the FL server that coordinates
    model training across hospital clients.
    
    Args:
        server_address: Address to bind (e.g., "0.0.0.0:8087")
        num_rounds: Number of training rounds
        min_clients: Minimum number of clients required
    
    Example:
        >>> start_demand_server(
        ...     server_address="0.0.0.0:8087",
        ...     num_rounds=5,
        ...     min_clients=2,
        ... )
    """
    # Override config if parameters provided
    if server_address:
        DemandServerConfig.SERVER_ADDRESS = server_address
    if num_rounds:
        DemandServerConfig.NUM_ROUNDS = num_rounds
    if min_clients:
        DemandServerConfig.MIN_FIT_CLIENTS = min_clients
        DemandServerConfig.MIN_EVALUATE_CLIENTS = min_clients
        DemandServerConfig.MIN_AVAILABLE_CLIENTS = min_clients
    
    # Log startup banner
    print_startup_banner()
    
    # Create strategy and config
    strategy = create_strategy()
    config = create_server_config()
    
    logger.info(f"Starting FL Demand Server on {DemandServerConfig.SERVER_ADDRESS}")
    logger.info(f"Configuration: {DemandServerConfig.NUM_ROUNDS} rounds, min {DemandServerConfig.MIN_FIT_CLIENTS} clients")
    
    try:
        # Start the server
        fl.server.start_server(
            server_address=DemandServerConfig.SERVER_ADDRESS,
            config=config,
            strategy=strategy,
            grpc_max_message_length=DemandServerConfig.GRPC_MAX_MESSAGE_LENGTH,
        )
        
        logger.info("FL Demand Server completed successfully")
        
        # Get final aggregation summary
        summary = strategy.get_aggregation_summary()
        logger.info(f"Final aggregation summary: {summary}")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


def print_startup_banner() -> None:
    """Print a startup banner with server information."""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║            FEDERATED LEARNING SERVER - DEMAND FORECASTING                    ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   Server Address:    {DemandServerConfig.SERVER_ADDRESS:<52} ║
    ║   Training Rounds:   {DemandServerConfig.NUM_ROUNDS:<52} ║
    ║   Min Clients:       {DemandServerConfig.MIN_FIT_CLIENTS:<52} ║
    ║   Strategy:          BestModelStrategy (Performance-Weighted FedAvg)        ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   DATA SOVEREIGNTY GUARANTEE:                                                ║
    ║   • Patient data NEVER leaves hospital networks                              ║
    ║   • Only model parameters (weights) are transmitted                          ║
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
    """Main entry point for the demand FL server."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Server for Hospital Demand Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with defaults (5 rounds, min 2 clients)
  python demand_server.py
  
  # Start with custom configuration
  python demand_server.py --rounds 10 --min-clients 3
  
  # Start on custom address
  python demand_server.py --address 0.0.0.0:9000

Environment Variables:
  FL_SERVER_ADDRESS       Server bind address (default: 0.0.0.0:8087)
  FL_NUM_ROUNDS          Number of training rounds (default: 5)
  FL_MIN_FIT_CLIENTS     Minimum clients for training (default: 2)
  FL_MIN_EVAL_CLIENTS    Minimum clients for evaluation (default: 2)
  FL_MODEL_SAVE_PATH     Path to save global models (default: /tmp/fl_demand_model)
        """,
    )
    
    parser.add_argument(
        "--address",
        type=str,
        default=None,
        help=f"Server address (default: {DemandServerConfig.SERVER_ADDRESS})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=f"Number of training rounds (default: {DemandServerConfig.NUM_ROUNDS})",
    )
    parser.add_argument(
        "--min-clients",
        type=int,
        default=None,
        help=f"Minimum number of clients (default: {DemandServerConfig.MIN_FIT_CLIENTS})",
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
    
    start_demand_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
    )


if __name__ == "__main__":
    main()
