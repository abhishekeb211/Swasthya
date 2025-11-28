"""
Triage & Acuity Agent - Federated Learning Client

This module implements the Flower (flwr) client for federated learning,
enabling privacy-preserving collaborative triage model training across
the hospital network.

================================================================================
FEDERATED LEARNING FOR TRIAGE: PRIVACY-PRESERVING MEDICAL AI
================================================================================

                    ┌─────────────────────────────────┐
                    │    FL Server (Port 8086)        │
                    │    fl-triage-server             │
                    │                                 │
                    │  • Aggregates XGBoost models    │
                    │  • Coordinates FL rounds        │
                    │  • Never sees patient data      │
                    └───────────────┬─────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
   │   Hospital A    │     │   Hospital B    │     │   Hospital C    │
   │   FL Client     │     │   FL Client     │     │   FL Client     │
   │                 │     │                 │     │                 │
   │ Local triage    │     │ Local triage    │     │ Local triage    │
   │ data stays here │     │ data stays here │     │ data stays here │
   │ Train locally   │     │ Train locally   │     │ Train locally   │
   │ Send gradients  │     │ Send gradients  │     │ Send gradients  │
   └─────────────────┘     └─────────────────┘     └─────────────────┘

WHY FEDERATED LEARNING FOR TRIAGE?
──────────────────────────────────

1. HIPAA COMPLIANCE:
   Patient data never leaves the hospital.
   Only model parameters (tree structures) are shared.

2. DIVERSE TRAINING DATA:
   Different hospitals see different patient populations.
   Urban trauma centers vs. rural community hospitals.
   Pediatric specialists vs. general emergency.
   FL combines insights from all without centralizing data.

3. RARE CONDITION DETECTION:
   A single hospital may see few cases of rare emergencies.
   FL aggregates patterns across the network.
   Better detection of uncommon but critical presentations.

4. CONTINUOUS IMPROVEMENT:
   As hospitals collect new triage outcomes, models improve.
   New patterns (e.g., pandemic symptoms) propagate to all.

================================================================================
"""

from __future__ import annotations

import logging
import os
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from config import settings
from model import TriageClassifier, create_sample_data


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

class TriageDataLoader:
    """
    Loads local triage data for federated training.
    
    In production, this would connect to the hospital's database.
    For development, it can generate synthetic data.
    """
    
    def __init__(self, use_synthetic: bool = False):
        """
        Initialize the data loader.
        
        Args:
            use_synthetic: If True, generate synthetic data instead of loading from DB
        """
        self.use_synthetic = use_synthetic
        self._train_data: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
        self._test_data: Optional[Tuple[pd.DataFrame, np.ndarray]] = None
    
    def load_data(
        self,
        test_split: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[Tuple[pd.DataFrame, np.ndarray], Tuple[pd.DataFrame, np.ndarray]]:
        """
        Load and split triage data.
        
        Args:
            test_split: Fraction of data for testing
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of ((X_train, y_train), (X_test, y_test))
        """
        if self.use_synthetic:
            # Generate synthetic data
            X, y = create_sample_data(n_samples=2000, random_state=random_state)
        else:
            # In production, load from database
            # For now, fall back to synthetic
            try:
                X, y = self._load_from_database()
            except Exception as e:
                logger.warning(f"Could not load from database, using synthetic: {e}")
                X, y = create_sample_data(n_samples=2000, random_state=random_state)
        
        # Split into train/test
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_split,
            stratify=y,
            random_state=random_state,
        )
        
        self._train_data = (X_train, y_train)
        self._test_data = (X_test, y_test)
        
        logger.info(
            f"Data loaded",
            extra={
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "class_distribution": dict(zip(*np.unique(y_train, return_counts=True))),
            }
        )
        
        return self._train_data, self._test_data
    
    def _load_from_database(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load real triage data from PostgreSQL."""
        from sqlalchemy import create_engine, text
        
        engine = create_engine(str(settings.database_url))
        
        query = text("""
            SELECT 
                heart_rate, systolic_bp, diastolic_bp,
                respiratory_rate, oxygen_saturation, temperature,
                gcs, age, gender, chief_complaint_category,
                pain_score, arrival_mode, acuity_level
            FROM triage_records
            WHERE acuity_level IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 10000
        """)
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        if df.empty:
            raise ValueError("No triage data found in database")
        
        y = df["acuity_level"].values
        X = df.drop(columns=["acuity_level"])
        
        return X, y
    
    @property
    def train_data(self) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        return self._train_data
    
    @property
    def test_data(self) -> Optional[Tuple[pd.DataFrame, np.ndarray]]:
        return self._test_data


# =============================================================================
# FLOWER NUMPY CLIENT
# =============================================================================

class TriageFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPy client for federated triage model training.
    
    This client:
    1. Receives global model parameters from FL server
    2. Trains locally on hospital's own triage data
    3. Sends updated parameters back for aggregation
    4. Evaluates global model on local test data
    """
    
    def __init__(
        self,
        hospital_id: str,
        use_synthetic: bool = False,
    ) -> None:
        """
        Initialize the FL client.
        
        Args:
            hospital_id: Unique identifier for this hospital
            use_synthetic: Use synthetic data for testing
        """
        self.hospital_id = hospital_id
        self.use_synthetic = use_synthetic
        
        # Initialize classifier
        self.classifier = TriageClassifier()
        
        # Load local data
        self.data_loader = TriageDataLoader(use_synthetic=use_synthetic)
        self.train_data, self.test_data = self.data_loader.load_data()
        
        logger.info(
            f"FL Client initialized for hospital: {hospital_id}",
            extra={
                "train_samples": len(self.train_data[0]) if self.train_data else 0,
                "test_samples": len(self.test_data[0]) if self.test_data else 0,
            }
        )
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Return current model parameters to the server.
        
        Returns:
            List of numpy arrays representing model parameters
        """
        if not self.classifier.is_fitted:
            # Train on local data first if no model exists
            if self.train_data:
                X_train, y_train = self.train_data
                self.classifier.fit(X_train, y_train)
        
        return self.classifier.get_booster_params()
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train model on local data after receiving global parameters.
        
        This is the core of federated learning:
        1. Apply global parameters (if provided)
        2. Train on local hospital data
        3. Return updated parameters and metrics
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
        
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        logger.info(f"Starting FL training round for {self.hospital_id}")
        
        if not self.train_data:
            logger.warning("No training data available")
            return [], 0, {"error": "no_data"}
        
        X_train, y_train = self.train_data
        
        try:
            # Apply global parameters if provided
            if parameters:
                try:
                    self.classifier.set_booster_params(parameters)
                    logger.info("Applied global parameters from server")
                except Exception as e:
                    logger.warning(f"Could not apply global params: {e}")
            
            # Get local epochs from config or settings
            local_epochs = int(config.get("local_epochs", settings.fl_local_epochs))
            
            # Train on local data
            # For XGBoost, we retrain with warm start behavior
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(self.test_data[0], self.test_data[1])] if self.test_data else None,
                verbose=False,
            )
            
            # Extract updated parameters
            updated_params = self.classifier.get_booster_params()
            
            # Compute local metrics
            metrics: Dict[str, Scalar] = {
                "hospital_id": self.hospital_id,
            }
            
            if self.test_data:
                X_test, y_test = self.test_data
                eval_metrics = self.classifier.evaluate(X_test, y_test)
                metrics.update({
                    "accuracy": float(eval_metrics["accuracy"]),
                    "macro_f1": float(eval_metrics["macro_f1"]),
                    "under_triage_rate": float(eval_metrics["under_triage_rate"]),
                })
            
            logger.info(
                f"FL training complete for {self.hospital_id}",
                extra=metrics,
            )
            
            return updated_params, len(X_train), metrics
            
        except Exception as e:
            logger.error(f"FL training failed: {e}")
            return [], 0, {"error": str(e)}
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate global model on local test data.
        
        This allows the FL server to assess model quality
        without accessing any hospital's data.
        
        Args:
            parameters: Global model parameters to evaluate
            config: Evaluation configuration
        
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        logger.info(f"Evaluating global model at {self.hospital_id}")
        
        if not self.test_data:
            return float("inf"), 0, {"error": "no_test_data"}
        
        X_test, y_test = self.test_data
        
        try:
            # Apply global parameters
            if parameters:
                self.classifier.set_booster_params(parameters)
            elif not self.classifier.is_fitted:
                # Need to train first
                if self.train_data:
                    X_train, y_train = self.train_data
                    self.classifier.fit(X_train, y_train)
            
            # Evaluate
            metrics = self.classifier.evaluate(X_test, y_test)
            
            # Use (1 - accuracy) as loss
            loss = 1.0 - metrics["accuracy"]
            
            result_metrics: Dict[str, Scalar] = {
                "accuracy": float(metrics["accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "under_triage_rate": float(metrics["under_triage_rate"]),
                "hospital_id": self.hospital_id,
            }
            
            logger.info(
                f"Evaluation complete at {self.hospital_id}",
                extra=result_metrics,
            )
            
            return loss, len(X_test), result_metrics
            
        except Exception as e:
            logger.error(f"FL evaluation failed: {e}")
            return float("inf"), 0, {"error": str(e)}


# =============================================================================
# CLIENT STARTUP
# =============================================================================

def start_fl_client(
    server_address: Optional[str] = None,
    hospital_id: Optional[str] = None,
    use_synthetic: bool = False,
) -> None:
    """
    Start the Federated Learning client.
    
    Connects to the FL server and participates in training rounds.
    The client will wait for instructions from the server.
    
    Args:
        server_address: FL server address (default from settings)
        hospital_id: Unique identifier for this hospital
        use_synthetic: Use synthetic data for testing
    """
    server_address = server_address or settings.fl_server_address
    hospital_id = hospital_id or settings.hospital_id
    
    logger.info(
        f"Starting FL client",
        extra={
            "server": server_address,
            "hospital_id": hospital_id,
            "use_synthetic": use_synthetic,
        }
    )
    
    # Create client
    client = TriageFlowerClient(
        hospital_id=hospital_id,
        use_synthetic=use_synthetic,
    )
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
    )


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    """
    Command-line interface for starting the FL client.
    
    Usage:
        python fl_run_client.py [--server ADDRESS] [--hospital-id ID] [--synthetic]
    
    Examples:
        # Connect to default server with synthetic data
        python fl_run_client.py --synthetic
        
        # Connect to specific server
        python fl_run_client.py --server fl-triage-server:8086 --hospital-id hospital_a
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Start FL client for triage model training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Start with synthetic data for testing
    python fl_run_client.py --synthetic
    
    # Connect to specific FL server
    python fl_run_client.py --server fl-triage-server:8086
    
    # Specify hospital identifier
    python fl_run_client.py --hospital-id memorial_hospital_west
        """
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help=f"FL server address (default: {settings.fl_server_address})",
    )
    parser.add_argument(
        "--hospital-id",
        type=str,
        default=os.environ.get("HOSPITAL_ID", settings.hospital_id),
        help="Unique hospital identifier",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing (no database required)",
    )
    
    args = parser.parse_args()
    
    # Banner
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║          TRIAGE & ACUITY - FEDERATED LEARNING CLIENT             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Privacy-preserving collaborative triage model training          ║
    ║                                                                  ║
    ║  • Patient data NEVER leaves your hospital                       ║
    ║  • Only model parameters are shared                              ║
    ║  • HIPAA-compliant federated learning                            ║
    ╠══════════════════════════════════════════════════════════════════╣""")
    print(f"    ║  Hospital ID: {args.hospital_id:<48} ║")
    print(f"    ║  FL Server:   {args.server or settings.fl_server_address:<48} ║")
    print(f"    ║  Data Mode:   {'Synthetic (Testing)' if args.synthetic else 'Production (Database)':<48} ║")
    print("""    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Start client
    start_fl_client(
        server_address=args.server,
        hospital_id=args.hospital_id,
        use_synthetic=args.synthetic,
    )
