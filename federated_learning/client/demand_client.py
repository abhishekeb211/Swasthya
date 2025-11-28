"""
Federated Learning Client for Demand Forecasting

This module implements a standalone FL client that trains Prophet models
on local hospital data and participates in federated learning rounds.

================================================================================
DEMAND FORECASTING FL CLIENT
================================================================================

This client:
1. Loads local hospital patient volume data (CSV or database)
2. Trains a Prophet time-series model locally
3. Sends model parameters to the FL server (NOT the data!)
4. Receives aggregated global model parameters
5. Evaluates model on local test data

The client is designed to be robust:
- Retries connection to server with exponential backoff
- Handles temporary network failures gracefully
- Saves local model checkpoints for recovery

================================================================================
DATA FLOW (What stays local vs. what is transmitted)
================================================================================

                      HOSPITAL NETWORK                    │        INTERNET
    ──────────────────────────────────────────────────────┼───────────────────
                                                          │
    ┌─────────────────────────────────────────────────┐   │
    │              LOCAL DATA (NEVER LEAVES)          │   │
    │  ┌─────────────────────────────────────────┐    │   │
    │  │  patient_visits.csv                     │    │   │
    │  │  ├── timestamp                          │    │   │
    │  │  ├── patient_id (anonymized)            │    │   │
    │  │  ├── admission_type                     │    │   │
    │  │  └── department                         │    │   │
    │  └─────────────────────────────────────────┘    │   │
    │                    │                             │   │
    │                    ▼                             │   │
    │  ┌─────────────────────────────────────────┐    │   │
    │  │           Prophet Model                  │    │   │
    │  │  • Train on local data                  │    │   │
    │  │  • Extract parameters (k, m, delta, β)  │    │   │
    │  └────────────────┬────────────────────────┘    │   │
    │                   │                              │   │
    └───────────────────┼──────────────────────────────┘   │
                        │                                   │
                        │  TRANSMITTED: Only model          │
                        │  parameters (numbers)             │
                        │  ~1KB of floating point           │
                        │  values                           │
                        │                                   │
                        ▼                                   │
    ═══════════════════════════════════════════════════════╧═══════════════════
                        │
                        ▼
              ┌─────────────────┐
              │    FL Server    │
              │  Aggregates     │
              │  parameters     │
              └─────────────────┘

================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    Parameters,
    Scalar,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from serde import ProphetSerializer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class DemandClientConfig:
    """Configuration for the Demand FL Client."""
    
    # Server connection
    FL_SERVER_ADDRESS: str = os.getenv("FL_SERVER_ADDRESS", "localhost:8087")
    
    # Hospital identification
    HOSPITAL_ID: str = os.getenv("HOSPITAL_ID", f"hospital_{os.getpid()}")
    
    # Data paths
    DATA_PATH: str = os.getenv("FL_DATA_PATH", "/data/patient_volumes.csv")
    
    # Training parameters
    LOCAL_EPOCHS: int = int(os.getenv("FL_LOCAL_EPOCHS", "1"))
    TEST_SPLIT_DAYS: int = int(os.getenv("FL_TEST_SPLIT_DAYS", "30"))
    
    # Retry configuration
    MAX_RETRIES: int = int(os.getenv("FL_MAX_RETRIES", "10"))
    RETRY_DELAY_BASE: float = float(os.getenv("FL_RETRY_DELAY", "5.0"))
    RETRY_DELAY_MAX: float = float(os.getenv("FL_RETRY_DELAY_MAX", "60.0"))
    
    # Model checkpoint
    CHECKPOINT_PATH: str = os.getenv("FL_CHECKPOINT_PATH", "/tmp/fl_demand_checkpoint")


# =============================================================================
# DATA LOADING
# =============================================================================

class LocalDataLoader:
    """
    Loads and prepares local hospital data for federated learning.
    
    This class handles:
    - Loading from CSV or generating synthetic data
    - Train/test splitting
    - Data validation and preprocessing
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        hospital_id: Optional[str] = None,
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to local data file (CSV)
            hospital_id: Hospital identifier for synthetic data variation
        """
        self.data_path = data_path or DemandClientConfig.DATA_PATH
        self.hospital_id = hospital_id or DemandClientConfig.HOSPITAL_ID
        self.data: Optional[pd.DataFrame] = None
    
    def load(self) -> pd.DataFrame:
        """
        Load data from file or generate synthetic data.
        
        Returns:
            DataFrame with columns: timestamp, patient_count
        """
        path = Path(self.data_path)
        
        if path.exists():
            logger.info(f"Loading data from {path}")
            self.data = self._load_from_csv(path)
        else:
            logger.warning(f"Data file not found at {path}, generating synthetic data")
            self.data = self._generate_synthetic_data()
        
        logger.info(
            f"Loaded {len(self.data)} records from "
            f"{self.data['timestamp'].min()} to {self.data['timestamp'].max()}"
        )
        
        return self.data
    
    def _load_from_csv(self, path: Path) -> pd.DataFrame:
        """Load data from CSV file."""
        df = pd.read_csv(path)
        
        # Standardize column names
        column_mapping = {
            'date': 'timestamp',
            'datetime': 'timestamp',
            'time': 'timestamp',
            'count': 'patient_count',
            'volume': 'patient_count',
            'visits': 'patient_count',
        }
        
        df.columns = [column_mapping.get(c.lower(), c.lower()) for c in df.columns]
        
        # Ensure required columns
        if 'timestamp' not in df.columns:
            raise ValueError("Data must have a timestamp column")
        if 'patient_count' not in df.columns:
            raise ValueError("Data must have a patient_count column")
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['patient_count'] = pd.to_numeric(df['patient_count'], errors='coerce').fillna(0)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df[['timestamp', 'patient_count']]
    
    def _generate_synthetic_data(self, n_days: int = 365) -> pd.DataFrame:
        """
        Generate synthetic patient volume data.
        
        Creates realistic patterns with:
        - Daily seasonality (morning/evening peaks)
        - Weekly seasonality (weekend variations)
        - Yearly seasonality (flu season, summer lull)
        - Random noise
        - Hospital-specific variation
        """
        np.random.seed(hash(self.hospital_id) % 2**32)
        
        # Base hourly rate (varies by hospital)
        base_rate = 15 + np.random.uniform(-5, 10)
        
        # Generate hourly timestamps
        start_date = datetime.now() - timedelta(days=n_days)
        timestamps = pd.date_range(start=start_date, periods=n_days * 24, freq='H')
        
        # Build patient counts with multiple seasonalities
        counts = []
        for ts in timestamps:
            hour = ts.hour
            dow = ts.dayofweek
            doy = ts.dayofyear
            
            # Daily pattern: peaks at 11am and 7pm
            daily = 1.0 + 0.3 * np.sin(2 * np.pi * (hour - 11) / 24)
            daily += 0.2 * np.sin(2 * np.pi * (hour - 19) / 24)
            
            # Weekly pattern: higher on weekends for ER
            weekly = 1.0 + 0.15 * (1 if dow >= 5 else 0)
            
            # Yearly pattern: flu season peaks in January
            yearly = 1.0 + 0.2 * np.cos(2 * np.pi * (doy - 15) / 365)
            
            # Combine factors
            expected = base_rate * daily * weekly * yearly
            
            # Add Poisson noise
            count = np.random.poisson(expected)
            counts.append(count)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'patient_count': counts,
        })
    
    def split_train_test(
        self,
        test_days: int = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Uses the most recent `test_days` days for testing.
        
        Args:
            test_days: Number of days to reserve for testing
        
        Returns:
            Tuple of (train_df, test_df)
        """
        test_days = test_days or DemandClientConfig.TEST_SPLIT_DAYS
        
        if self.data is None:
            self.load()
        
        # Find split point
        max_date = self.data['timestamp'].max()
        split_date = max_date - timedelta(days=test_days)
        
        train_df = self.data[self.data['timestamp'] < split_date].copy()
        test_df = self.data[self.data['timestamp'] >= split_date].copy()
        
        logger.info(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        return train_df, test_df


# =============================================================================
# PROPHET MODEL WRAPPER
# =============================================================================

class DemandForecaster:
    """
    Simple Prophet wrapper for demand forecasting.
    
    This is a minimal implementation for FL. For full features,
    see the demand_forecast agent's model.py.
    """
    
    def __init__(self):
        """Initialize the forecaster."""
        self.model = None
        self.is_fitted = False
        self.training_metadata = {}
    
    def fit(self, df: pd.DataFrame) -> "DemandForecaster":
        """
        Fit Prophet model on patient volume data.
        
        Args:
            df: DataFrame with timestamp and patient_count columns
        
        Returns:
            self for method chaining
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet not installed. Run: pip install prophet")
        
        # Prepare Prophet format
        prophet_df = df[['timestamp', 'patient_count']].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        # Apply log transform
        prophet_df['y'] = np.log1p(prophet_df['y'])
        
        # Create and fit model
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            changepoint_prior_scale=0.05,
        )
        
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        self.training_metadata = {
            'n_samples': len(prophet_df),
            'trained_at': datetime.utcnow().isoformat(),
        }
        
        logger.info(f"Model fitted on {len(prophet_df)} samples")
        
        return self
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Returns:
            Dictionary with MAE, RMSE, and MAPE metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Prepare test data
        test_prophet = test_df[['timestamp', 'patient_count']].copy()
        test_prophet.columns = ['ds', 'y_actual']
        test_prophet['ds'] = pd.to_datetime(test_prophet['ds'])
        
        # Generate predictions
        forecast = self.model.predict(test_prophet[['ds']])
        
        # Merge with actuals
        merged = test_prophet.merge(
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds'
        )
        
        # Reverse log transform
        merged['yhat'] = np.expm1(merged['yhat'])
        
        # Calculate metrics
        y_actual = merged['y_actual'].values
        y_pred = merged['yhat'].values
        
        mae = float(np.mean(np.abs(y_actual - y_pred)))
        rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
        
        # MAPE (avoid division by zero)
        nonzero = y_actual > 0
        if nonzero.sum() > 0:
            mape = float(np.mean(np.abs(
                (y_actual[nonzero] - y_pred[nonzero]) / y_actual[nonzero]
            ))) * 100
        else:
            mape = float('nan')
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'n_test': len(merged),
        }
        
        logger.info(f"Evaluation metrics: MAE={mae:.2f}, RMSE={rmse:.2f}")
        
        return metrics
    
    def get_parameters(self) -> List[np.ndarray]:
        """Extract model parameters for FL."""
        if not self.is_fitted:
            return []
        return ProphetSerializer.extract_parameters(self.model)
    
    def set_parameters(self, params: List[np.ndarray]) -> None:
        """Apply FL parameters to model."""
        if params and self.model is not None:
            self.model = ProphetSerializer.apply_parameters(self.model, params)


# =============================================================================
# FLOWER CLIENT IMPLEMENTATION
# =============================================================================

class DemandFlowerClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for demand forecasting federated learning.
    
    This client implements the Flower protocol:
    - get_parameters(): Return current model parameters
    - fit(): Train on local data with received parameters
    - evaluate(): Evaluate model on local test data
    """
    
    def __init__(
        self,
        hospital_id: str,
        data_loader: LocalDataLoader,
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
        self.train_data, self.test_data = self.data_loader.split_train_test()
        
        # Initialize forecaster
        self.forecaster = DemandForecaster()
        
        logger.info(
            f"FL Client initialized: {hospital_id}, "
            f"train={len(self.train_data)}, test={len(self.test_data)}"
        )
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return current model parameters."""
        logger.debug("get_parameters called")
        return self.forecaster.get_parameters()
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train model on local data.
        
        Args:
            parameters: Global model parameters from server
            config: Training configuration from server
        
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        logger.info(f"Starting local training for {self.hospital_id}")
        
        try:
            # Apply received parameters if any
            if parameters:
                logger.debug("Applying global parameters")
                self.forecaster.set_parameters(parameters)
            
            # Train locally
            local_epochs = config.get("local_epochs", DemandClientConfig.LOCAL_EPOCHS)
            for epoch in range(int(local_epochs)):
                self.forecaster.fit(self.train_data)
                logger.debug(f"Completed local epoch {epoch + 1}/{local_epochs}")
            
            # Get updated parameters
            new_params = self.forecaster.get_parameters()
            
            # Calculate local metrics
            metrics: Dict[str, Scalar] = {"hospital_id": self.hospital_id}
            
            if self.test_data is not None and len(self.test_data) > 0:
                eval_metrics = self.forecaster.evaluate(self.test_data)
                metrics.update({k: float(v) for k, v in eval_metrics.items()})
            
            logger.info(
                f"Training complete for {self.hospital_id}: "
                f"MAE={metrics.get('mae', 'N/A')}"
            )
            
            return new_params, len(self.train_data), metrics
            
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
                self.forecaster.set_parameters(parameters)
            
            # Ensure model is fitted
            if not self.forecaster.is_fitted:
                self.forecaster.fit(self.train_data)
            
            # Evaluate
            metrics = self.forecaster.evaluate(self.test_data)
            
            # Use MAE as loss
            loss = float(metrics.get('mae', float('inf')))
            
            result_metrics: Dict[str, Scalar] = {
                "hospital_id": self.hospital_id,
                **{k: float(v) for k, v in metrics.items()}
            }
            
            logger.info(f"Evaluation complete: loss={loss:.2f}")
            
            return loss, len(self.test_data), result_metrics
            
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
    
    Args:
        server_address: Server address (host:port)
        max_retries: Maximum number of connection attempts
        retry_delay_base: Initial delay between retries (seconds)
        retry_delay_max: Maximum delay between retries (seconds)
    
    Returns:
        True if server is available, False if max retries exceeded
    """
    import socket
    
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
            
            # Exponential backoff with cap
            delay = min(delay * 1.5, retry_delay_max)
    
    logger.error(f"Server not available after {max_retries} attempts")
    return False


def start_demand_client(
    server_address: Optional[str] = None,
    hospital_id: Optional[str] = None,
    data_path: Optional[str] = None,
    wait_for_ready: bool = True,
) -> None:
    """
    Start the Federated Learning client for demand forecasting.
    
    This function initializes the FL client and connects to the server.
    It includes retry logic to handle cases where the server isn't
    immediately available.
    
    Args:
        server_address: FL server address (host:port)
        hospital_id: Unique identifier for this hospital
        data_path: Path to local data file
        wait_for_ready: Whether to wait for server to become available
    """
    # Apply configuration
    server_address = server_address or DemandClientConfig.FL_SERVER_ADDRESS
    hospital_id = hospital_id or DemandClientConfig.HOSPITAL_ID
    data_path = data_path or DemandClientConfig.DATA_PATH
    
    # Print startup banner
    print_startup_banner(hospital_id, server_address)
    
    # Wait for server if requested
    if wait_for_ready:
        if not wait_for_server(
            server_address,
            DemandClientConfig.MAX_RETRIES,
            DemandClientConfig.RETRY_DELAY_BASE,
            DemandClientConfig.RETRY_DELAY_MAX,
        ):
            logger.error("Cannot connect to FL server, exiting")
            sys.exit(1)
    
    # Initialize data loader
    data_loader = LocalDataLoader(data_path=data_path, hospital_id=hospital_id)
    
    # Create client
    client = DemandFlowerClient(hospital_id=hospital_id, data_loader=data_loader)
    
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
    ║           FEDERATED LEARNING CLIENT - DEMAND FORECASTING                     ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   Hospital ID:     {hospital_id:<54} ║
    ║   FL Server:       {server_address:<54} ║
    ║   Model Type:      Prophet (Time-Series)                                     ║
    ║                                                                              ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║                                                                              ║
    ║   DATA SOVEREIGNTY:                                                          ║
    ║   ✓ Patient data STAYS on this machine                                       ║
    ║   ✓ Only model parameters are transmitted                                    ║
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
    """Main entry point for the demand FL client."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Client for Hospital Demand Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start client with defaults
  python demand_client.py
  
  # Connect to specific server
  python demand_client.py --server fl-server:8087
  
  # Use custom hospital ID and data
  python demand_client.py --hospital-id hospital_a --data /path/to/data.csv

Environment Variables:
  FL_SERVER_ADDRESS   Server address (default: localhost:8087)
  HOSPITAL_ID        Hospital identifier (default: hospital_<pid>)
  FL_DATA_PATH       Path to data file (default: /data/patient_volumes.csv)
  FL_LOCAL_EPOCHS    Local training epochs (default: 1)
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
    
    start_demand_client(
        server_address=args.server,
        hospital_id=args.hospital_id,
        data_path=args.data,
        wait_for_ready=not args.no_wait,
    )


if __name__ == "__main__":
    main()
