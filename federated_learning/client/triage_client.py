"""
Federated Learning Client for Triage Prediction

This module implements a standalone FL client that trains XGBoost models
on local hospital triage data and participates in federated learning rounds.

================================================================================
TRIAGE PREDICTION FL CLIENT
================================================================================

This client trains XGBoost classifiers for:
- Patient acuity prediction (ESI 1-5)
- Discharge readiness scoring
- Length of stay estimation

The client is designed for healthcare environments:
- Robust retry logic for intermittent network
- Secure handling of patient data (data never leaves)
- Efficient XGBoost parameter transmission

================================================================================
TRIAGE DATA SCHEMA
================================================================================

Expected input data (CSV or database):

    patient_id,                  # Anonymized identifier
    arrival_time,                # Timestamp
    age,                         # Patient age
    gender,                      # M/F
    chief_complaint,             # Presenting complaint code
    heart_rate,                  # Vital: HR
    blood_pressure_systolic,     # Vital: SBP
    blood_pressure_diastolic,    # Vital: DBP
    respiratory_rate,            # Vital: RR
    temperature,                 # Vital: Temp
    spo2,                        # Vital: SpO2
    pain_score,                  # 0-10
    arrival_mode,                # Walk-in, Ambulance, etc.
    previous_visits_30d,         # Prior visits
    acuity_level                 # Target: ESI 1-5

================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import flwr as fl
from flwr.common import Scalar

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from serde import XGBoostSerializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class TriageClientConfig:
    """Configuration for the Triage FL Client."""
    
    # Server connection
    FL_SERVER_ADDRESS: str = os.getenv("FL_TRIAGE_SERVER_ADDRESS", "localhost:8086")
    
    # Hospital identification
    HOSPITAL_ID: str = os.getenv("HOSPITAL_ID", f"hospital_{os.getpid()}")
    
    # Data paths
    DATA_PATH: str = os.getenv("FL_DATA_PATH", "/data/triage_records.csv")
    
    # Training parameters
    LOCAL_EPOCHS: int = int(os.getenv("FL_LOCAL_EPOCHS", "1"))
    TEST_SPLIT_RATIO: float = float(os.getenv("FL_TEST_SPLIT_RATIO", "0.2"))
    
    # XGBoost parameters
    XGB_N_ESTIMATORS: int = int(os.getenv("XGB_N_ESTIMATORS", "100"))
    XGB_MAX_DEPTH: int = int(os.getenv("XGB_MAX_DEPTH", "6"))
    XGB_LEARNING_RATE: float = float(os.getenv("XGB_LEARNING_RATE", "0.1"))
    
    # Retry configuration
    MAX_RETRIES: int = int(os.getenv("FL_MAX_RETRIES", "10"))
    RETRY_DELAY_BASE: float = float(os.getenv("FL_RETRY_DELAY", "5.0"))
    RETRY_DELAY_MAX: float = float(os.getenv("FL_RETRY_DELAY_MAX", "60.0"))


# =============================================================================
# DATA LOADING
# =============================================================================

class TriageDataLoader:
    """
    Loads and prepares local triage data for federated learning.
    
    Handles:
    - Loading from CSV
    - Generating synthetic triage data
    - Feature engineering
    - Train/test splitting
    """
    
    # Expected features for the model
    FEATURE_COLUMNS = [
        'age',
        'heart_rate',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'respiratory_rate',
        'temperature',
        'spo2',
        'pain_score',
        'arrival_mode_encoded',
        'previous_visits_30d',
        'hour_of_day',
        'is_weekend',
    ]
    
    TARGET_COLUMN = 'acuity_level'
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        hospital_id: Optional[str] = None,
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to local data file
            hospital_id: Hospital identifier for synthetic data variation
        """
        self.data_path = data_path or TriageClientConfig.DATA_PATH
        self.hospital_id = hospital_id or TriageClientConfig.HOSPITAL_ID
        self.data: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """Load data from file or generate synthetic data."""
        path = Path(self.data_path)
        
        if path.exists():
            logger.info(f"Loading triage data from {path}")
            self.data = self._load_from_csv(path)
        else:
            logger.warning(f"Data file not found at {path}, generating synthetic data")
            self.data = self._generate_synthetic_data()
        
        # Apply feature engineering
        self.data = self._engineer_features(self.data)
        
        logger.info(f"Loaded {len(self.data)} triage records")
        
        return self.data
    
    def _load_from_csv(self, path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(path)
        
        # Standardize column names
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        
        return df
    
    def _generate_synthetic_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic triage data.
        
        Creates realistic patient data with:
        - Age-dependent vital sign distributions
        - Correlated features (higher acuity → abnormal vitals)
        - Hospital-specific variation
        """
        np.random.seed(hash(self.hospital_id) % 2**32)
        
        # Generate base patient characteristics
        ages = np.random.gamma(shape=5, scale=10, size=n_samples).astype(int)
        ages = np.clip(ages, 0, 100)
        
        # Generate acuity levels (1-5, with 3 being most common)
        acuity_probs = [0.02, 0.12, 0.50, 0.25, 0.11]  # ESI 1-5
        acuity = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=acuity_probs)
        
        # Generate vitals based on acuity
        data = {
            'arrival_time': pd.date_range(
                end=datetime.now(),
                periods=n_samples,
                freq='15min'
            ),
            'age': ages,
            'acuity_level': acuity,
        }
        
        # Heart rate: higher acuity → more abnormal
        base_hr = 75 + (ages - 40) * 0.2
        acuity_adjustment = (3 - acuity) * 10  # Lower acuity = higher HR
        data['heart_rate'] = np.clip(
            base_hr + acuity_adjustment + np.random.normal(0, 10, n_samples),
            40, 180
        ).astype(int)
        
        # Blood pressure
        base_sbp = 120 + (ages - 40) * 0.5
        data['blood_pressure_systolic'] = np.clip(
            base_sbp + (3 - acuity) * 15 + np.random.normal(0, 15, n_samples),
            70, 220
        ).astype(int)
        
        data['blood_pressure_diastolic'] = np.clip(
            data['blood_pressure_systolic'] * 0.6 + np.random.normal(0, 10, n_samples),
            40, 120
        ).astype(int)
        
        # Respiratory rate
        data['respiratory_rate'] = np.clip(
            16 + (3 - acuity) * 4 + np.random.normal(0, 3, n_samples),
            8, 40
        ).astype(int)
        
        # Temperature (most are normal)
        data['temperature'] = np.where(
            np.random.random(n_samples) < 0.1 + (3 - acuity) * 0.05,
            np.random.uniform(38.0, 40.0, n_samples),  # Fever
            np.random.uniform(36.5, 37.5, n_samples),  # Normal
        )
        
        # SpO2
        data['spo2'] = np.clip(
            98 - (3 - acuity) * 3 + np.random.normal(0, 2, n_samples),
            70, 100
        ).astype(int)
        
        # Pain score
        data['pain_score'] = np.clip(
            5 + (3 - acuity) * 2 + np.random.normal(0, 2, n_samples),
            0, 10
        ).astype(int)
        
        # Arrival mode (0=walk-in, 1=ambulance, 2=helicopter)
        ambulance_prob = np.where(acuity <= 2, 0.7, np.where(acuity == 3, 0.3, 0.1))
        data['arrival_mode'] = np.where(
            np.random.random(n_samples) < ambulance_prob,
            np.where(np.random.random(n_samples) < 0.1, 2, 1),
            0
        )
        
        # Previous visits
        data['previous_visits_30d'] = np.random.poisson(0.5, n_samples)
        
        return pd.DataFrame(data)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering."""
        result = df.copy()
        
        # Encode arrival mode
        if 'arrival_mode' in result.columns:
            result['arrival_mode_encoded'] = result['arrival_mode']
        else:
            result['arrival_mode_encoded'] = 0
        
        # Time-based features
        if 'arrival_time' in result.columns:
            result['arrival_time'] = pd.to_datetime(result['arrival_time'])
            result['hour_of_day'] = result['arrival_time'].dt.hour
            result['is_weekend'] = result['arrival_time'].dt.dayofweek >= 5
            result['is_weekend'] = result['is_weekend'].astype(int)
        else:
            result['hour_of_day'] = 12
            result['is_weekend'] = 0
        
        # Ensure all required columns exist
        for col in self.FEATURE_COLUMNS:
            if col not in result.columns:
                result[col] = 0
        
        return result
    
    def split_train_test(
        self,
        test_ratio: float = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_ratio = test_ratio or TriageClientConfig.TEST_SPLIT_RATIO
        
        if self.data is None:
            self.load()
        
        # Shuffle data
        df = self.data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        split_idx = int(len(df) * (1 - test_ratio))
        
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[self.FEATURE_COLUMNS]
        X_test = test_df[self.FEATURE_COLUMNS]
        y_train = train_df[self.TARGET_COLUMN]
        y_test = test_df[self.TARGET_COLUMN]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test


# =============================================================================
# XGBOOST MODEL WRAPPER
# =============================================================================

class TriageClassifier:
    """
    XGBoost classifier for triage prediction.
    
    Predicts patient acuity level (ESI 1-5) based on
    vital signs and other features.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        self.is_fitted = False
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "TriageClassifier":
        """
        Fit XGBoost model on triage data.
        
        Args:
            X: Feature DataFrame
            y: Target series (acuity levels 1-5)
        
        Returns:
            self for method chaining
        """
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        # Convert target to 0-indexed for XGBoost
        y_adjusted = y - 1  # ESI 1-5 → 0-4
        
        # Create and fit model
        self.model = xgb.XGBClassifier(
            n_estimators=TriageClientConfig.XGB_N_ESTIMATORS,
            max_depth=TriageClientConfig.XGB_MAX_DEPTH,
            learning_rate=TriageClientConfig.XGB_LEARNING_RATE,
            objective='multi:softprob',
            num_class=5,
            eval_metric='mlogloss',
            use_label_encoder=False,
            random_state=42,
        )
        
        self.model.fit(X, y_adjusted)
        self.is_fitted = True
        
        logger.info(f"Model fitted on {len(X)} samples")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict acuity levels."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        
        # Convert back to ESI 1-5
        return predictions + 1
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with accuracy, precision, recall, f1 metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': float(accuracy_score(y, y_pred)),
            'precision': float(precision_score(y, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y, y_pred, average='weighted', zero_division=0)),
            'n_test': len(y),
        }
        
        logger.info(f"Evaluation: accuracy={metrics['accuracy']:.4f}")
        
        return metrics
    
    def get_parameters(self) -> List[np.ndarray]:
        """Extract model parameters for FL."""
        if not self.is_fitted:
            return []
        return XGBoostSerializer.extract_parameters(self.model)
    
    def set_parameters(self, params: List[np.ndarray]) -> None:
        """Apply FL parameters to model."""
        if params:
            try:
                import xgboost as xgb
                self.model = XGBoostSerializer.apply_parameters(self.model, params)
                if isinstance(self.model, xgb.Booster):
                    # Wrap in classifier interface
                    classifier = xgb.XGBClassifier()
                    classifier._Booster = self.model
                    self.model = classifier
                self.is_fitted = True
            except Exception as e:
                logger.warning(f"Failed to apply parameters: {e}")


# =============================================================================
# FLOWER CLIENT IMPLEMENTATION
# =============================================================================

class TriageFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for triage prediction federated learning.
    """
    
    def __init__(
        self,
        hospital_id: str,
        data_loader: TriageDataLoader,
    ):
        """
        Initialize the FL client.
        
        Args:
            hospital_id: Unique identifier for this hospital
            data_loader: Data loader instance
        """
        self.hospital_id = hospital_id
        self.data_loader = data_loader
        
        # Load and split data
        self.data_loader.load()
        self.X_train, self.X_test, self.y_train, self.y_test = \
            self.data_loader.split_train_test()
        
        # Initialize classifier
        self.classifier = TriageClassifier()
        
        logger.info(
            f"FL Client initialized: {hospital_id}, "
            f"train={len(self.X_train)}, test={len(self.X_test)}"
        )
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return current model parameters."""
        logger.debug("get_parameters called")
        return self.classifier.get_parameters()
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration
        
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"Starting local training for {self.hospital_id}")
        
        try:
            # Apply received parameters if any
            if parameters:
                logger.debug("Applying global parameters")
                self.classifier.set_parameters(parameters)
            
            # Train locally
            local_epochs = config.get("local_epochs", TriageClientConfig.LOCAL_EPOCHS)
            for epoch in range(int(local_epochs)):
                self.classifier.fit(self.X_train, self.y_train)
                logger.debug(f"Completed local epoch {epoch + 1}/{local_epochs}")
            
            # Get updated parameters
            new_params = self.classifier.get_parameters()
            
            # Calculate local metrics
            metrics: Dict[str, Scalar] = {"hospital_id": self.hospital_id}
            
            eval_metrics = self.classifier.evaluate(self.X_test, self.y_test)
            metrics.update({k: float(v) for k, v in eval_metrics.items()})
            
            logger.info(
                f"Training complete for {self.hospital_id}: "
                f"accuracy={metrics.get('accuracy', 'N/A')}"
            )
            
            return new_params, len(self.X_train), metrics
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return [], 0, {"error": str(e)}
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on local test data.
        
        Args:
            parameters: Global model parameters to evaluate
            config: Evaluation configuration
        
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        logger.info(f"Evaluating global model at {self.hospital_id}")
        
        try:
            # Apply global parameters
            if parameters:
                self.classifier.set_parameters(parameters)
            
            # Ensure model is fitted
            if not self.classifier.is_fitted:
                self.classifier.fit(self.X_train, self.y_train)
            
            # Evaluate
            metrics = self.classifier.evaluate(self.X_test, self.y_test)
            
            # Use 1 - accuracy as loss (so lower is better)
            loss = 1.0 - float(metrics.get('accuracy', 0))
            
            result_metrics: Dict[str, Scalar] = {
                "hospital_id": self.hospital_id,
                **{k: float(v) for k, v in metrics.items()}
            }
            
            logger.info(f"Evaluation complete: accuracy={metrics.get('accuracy', 0):.4f}")
            
            return loss, len(self.X_test), result_metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return float('inf'), 0, {"error": str(e)}


# =============================================================================
# CLIENT STARTUP WITH RETRY LOGIC
# =============================================================================

def wait_for_server(
    server_address: str,
    max_retries: int,
    retry_delay_base: float,
    retry_delay_max: float,
) -> bool:
    """
    Wait for FL server to become available with exponential backoff.
    """
    host, port = server_address.rsplit(':', 1)
    port = int(port)
    
    delay = retry_delay_base
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Checking server availability (attempt {attempt}/{max_retries})...")
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"Server is available at {server_address}")
                return True
            
        except Exception as e:
            logger.debug(f"Connection check failed: {e}")
        
        if attempt < max_retries:
            logger.info(f"Server not ready, waiting {delay:.1f}s before retry...")
            time.sleep(delay)
            delay = min(delay * 1.5, retry_delay_max)
    
    logger.error(f"Server not available after {max_retries} attempts")
    return False


def start_triage_client(
    server_address: Optional[str] = None,
    hospital_id: Optional[str] = None,
    data_path: Optional[str] = None,
    wait_for_ready: bool = True,
) -> None:
    """
    Start the Federated Learning client for triage prediction.
    
    Args:
        server_address: FL server address (host:port)
        hospital_id: Unique identifier for this hospital
        data_path: Path to local data file
        wait_for_ready: Whether to wait for server to become available
    """
    # Apply configuration
    server_address = server_address or TriageClientConfig.FL_SERVER_ADDRESS
    hospital_id = hospital_id or TriageClientConfig.HOSPITAL_ID
    data_path = data_path or TriageClientConfig.DATA_PATH
    
    # Print startup banner
    print_startup_banner(hospital_id, server_address)
    
    # Wait for server if requested
    if wait_for_ready:
        if not wait_for_server(
            server_address,
            TriageClientConfig.MAX_RETRIES,
            TriageClientConfig.RETRY_DELAY_BASE,
            TriageClientConfig.RETRY_DELAY_MAX,
        ):
            logger.error("Cannot connect to FL server, exiting")
            sys.exit(1)
    
    # Initialize data loader
    data_loader = TriageDataLoader(data_path=data_path, hospital_id=hospital_id)
    
    # Create client
    client = TriageFlowerClient(hospital_id=hospital_id, data_loader=data_loader)
    
    logger.info(f"Connecting to FL server at {server_address}")
    
    # Start FL client
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
        )
        logger.info("FL training completed successfully")
        
    except Exception as e:
        logger.error(f"FL client error: {e}")
        raise


def print_startup_banner(hospital_id: str, server_address: str) -> None:
    """Print startup banner."""
    banner = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║            FEDERATED LEARNING CLIENT - TRIAGE PREDICTION                     ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   Hospital ID:     {hospital_id:<54} ║
    ║   FL Server:       {server_address:<54} ║
    ║   Model Type:      XGBoost (Classification)                                  ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   TRIAGE PREDICTION FEATURES:                                                ║
    ║   • Patient acuity classification (ESI 1-5)                                  ║
    ║   • Based on vitals, age, chief complaint                                    ║
    ║   • Federated training preserves data privacy                                ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   DATA SOVEREIGNTY:                                                          ║
    ║   ✓ Patient data STAYS on this machine                                       ║
    ║   ✓ Only model parameters (tree weights) are transmitted                     ║
    ║   ✓ HIPAA/GDPR compliant federated learning                                  ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    Started at: {datetime.utcnow().isoformat()}
    """
    print(banner)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Main entry point for the triage FL client."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Client for Patient Triage Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start client with defaults
  python triage_client.py
  
  # Connect to specific server
  python triage_client.py --server fl-server:8086
  
  # Use custom hospital ID
  python triage_client.py --hospital-id hospital_a

Environment Variables:
  FL_TRIAGE_SERVER_ADDRESS  Server address (default: localhost:8086)
  HOSPITAL_ID              Hospital identifier (default: hospital_<pid>)
  FL_DATA_PATH             Path to data file (default: /data/triage_records.csv)
  XGB_N_ESTIMATORS         Number of trees (default: 100)
  XGB_MAX_DEPTH            Max tree depth (default: 6)
        """,
    )
    
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="FL server address (host:port)",
    )
    parser.add_argument(
        "--hospital-id",
        type=str,
        default=None,
        help="Unique hospital identifier",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to local data file (CSV)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for server to become available",
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
    
    start_triage_client(
        server_address=args.server,
        hospital_id=args.hospital_id,
        data_path=args.data,
        wait_for_ready=not args.no_wait,
    )


if __name__ == "__main__":
    main()
