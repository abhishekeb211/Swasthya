"""
Demand Forecast Agent - Federated Learning Client

This module implements the Flower (flwr) client for federated learning,
enabling privacy-preserving collaborative model training across the
hospital network.

================================================================================
FEDERATED LEARNING ARCHITECTURE FOR HOSPITAL AI
================================================================================

                    ┌─────────────────────────────┐
                    │    FL Server (Port 8087)    │
                    │   fl-demand-server          │
                    │                             │
                    │  • Aggregates model updates │
                    │  • Coordinates FL rounds    │
                    │  • No access to raw data    │
                    └─────────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │   Hospital A    │   │   Hospital B    │   │   Hospital C    │
   │   FL Client     │   │   FL Client     │   │   FL Client     │
   │                 │   │                 │   │                 │
   │ • Local data    │   │ • Local data    │   │ • Local data    │
   │ • Local train   │   │ • Local train   │   │ • Local train   │
   │ • Send weights  │   │ • Send weights  │   │ • Send weights  │
   └─────────────────┘   └─────────────────┘   └─────────────────┘

PRIVACY GUARANTEES:
───────────────────
1. Raw patient data NEVER leaves the hospital
2. Only model parameters (Prophet coefficients) are shared
3. Aggregation uses FedAvg - individual updates are anonymized
4. Differential privacy can be added for extra protection

WHY FEDERATED LEARNING FOR DEMAND FORECASTING?
──────────────────────────────────────────────
1. BETTER GENERALIZATION: Learn patterns across diverse populations
2. RARE EVENT DETECTION: Small hospitals may rarely see certain patterns
3. REGULATORY COMPLIANCE: HIPAA-friendly - no data centralization
4. NETWORK EFFECTS: Each hospital benefits from collective intelligence

================================================================================
"""

from __future__ import annotations

import logging
import json
import pickle
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
from model import DemandForecaster
from train import DataLoader, split_train_test

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ProphetParameterExtractor:
    """
    Extracts and reconstructs Prophet model parameters for federated learning.
    
    Prophet uses Stan under the hood, and its parameters include:
    - Trend parameters (k, m, delta)
    - Seasonality Fourier coefficients
    - Holiday effects
    
    For FL, we extract these as numpy arrays, aggregate them server-side,
    and reconstruct updated models client-side.
    """
    
    @staticmethod
    def extract_parameters(forecaster: DemandForecaster) -> List[np.ndarray]:
        """
        Extract trainable parameters from a fitted Prophet model.
        
        Args:
            forecaster: Fitted DemandForecaster instance
        
        Returns:
            List of numpy arrays containing model parameters
        """
        if not forecaster.is_fitted:
            raise RuntimeError("Cannot extract parameters from unfitted model")
        
        model = forecaster.model
        params = []
        
        # Extract trend parameters
        if hasattr(model, 'params') and model.params:
            # k (growth rate)
            if 'k' in model.params:
                params.append(np.array([model.params['k']]))
            
            # m (offset)
            if 'm' in model.params:
                params.append(np.array([model.params['m']]))
            
            # delta (changepoint adjustments)
            if 'delta' in model.params:
                params.append(np.array(model.params['delta']))
            
            # beta (seasonality coefficients)
            if 'beta' in model.params:
                params.append(np.array(model.params['beta']))
        
        # If no params extracted, return the serialized model as fallback
        if not params:
            # Serialize model and convert to array
            model_json = forecaster.serialize()
            params = [np.frombuffer(model_json.encode('utf-8'), dtype=np.uint8)]
            logger.warning("Using serialized model fallback for FL parameters")
        
        return params
    
    @staticmethod
    def apply_parameters(
        forecaster: DemandForecaster,
        parameters: List[np.ndarray],
    ) -> DemandForecaster:
        """
        Apply aggregated parameters to a Prophet model.
        
        Args:
            forecaster: DemandForecaster instance (fitted or unfitted)
            parameters: List of numpy arrays from FL server
        
        Returns:
            Updated DemandForecaster instance
        """
        # Check if this is a serialized model fallback
        if len(parameters) == 1 and parameters[0].dtype == np.uint8:
            model_json = parameters[0].tobytes().decode('utf-8')
            return DemandForecaster.deserialize(model_json)
        
        # Apply individual parameters
        model = forecaster.model
        param_idx = 0
        
        if hasattr(model, 'params') and model.params:
            if 'k' in model.params and param_idx < len(parameters):
                model.params['k'] = float(parameters[param_idx][0])
                param_idx += 1
            
            if 'm' in model.params and param_idx < len(parameters):
                model.params['m'] = float(parameters[param_idx][0])
                param_idx += 1
            
            if 'delta' in model.params and param_idx < len(parameters):
                model.params['delta'] = parameters[param_idx].tolist()
                param_idx += 1
            
            if 'beta' in model.params and param_idx < len(parameters):
                model.params['beta'] = parameters[param_idx].tolist()
                param_idx += 1
        
        return forecaster


class DemandForecastClient(fl.client.Client):
    """
    Flower client for federated demand forecasting.
    
    This client participates in federated learning rounds:
    1. Receives global model parameters from server
    2. Trains locally on hospital's own data
    3. Sends updated parameters back to server
    4. Evaluates model on local holdout set
    
    The client uses local data that NEVER leaves the hospital.
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
            use_synthetic: Use synthetic data for demo/testing
        """
        self.hospital_id = hospital_id
        self.use_synthetic = use_synthetic
        
        self.forecaster: Optional[DemandForecaster] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        self.parameter_extractor = ProphetParameterExtractor()
        
        # Load local data
        self._load_local_data()
        
        logger.info(
            f"FL Client initialized for hospital: {hospital_id}",
            extra={
                "train_samples": len(self.train_data) if self.train_data is not None else 0,
                "test_samples": len(self.test_data) if self.test_data is not None else 0,
            }
        )
    
    def _load_local_data(self) -> None:
        """Load local hospital data for federated training."""
        loader = DataLoader()
        
        try:
            if self.use_synthetic:
                # Generate hospital-specific synthetic data
                df = loader.generate_synthetic_data(
                    n_days=365,
                    base_hourly_rate=20.0 + np.random.uniform(-5, 5),
                )
            else:
                df = loader.load_aggregated_volume()
            
            if not df.empty:
                self.train_data, self.test_data = split_train_test(df, test_days=30)
            else:
                logger.warning("No data available for FL training")
                
        finally:
            loader.close()
    
    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Return client properties to the server."""
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={
                "hospital_id": self.hospital_id,
                "train_samples": len(self.train_data) if self.train_data is not None else 0,
                "test_samples": len(self.test_data) if self.test_data is not None else 0,
            },
        )
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return current model parameters to the server.
        
        Called by the server to initialize or inspect client models.
        """
        if self.forecaster is None or not self.forecaster.is_fitted:
            # Return empty parameters if no model yet
            return GetParametersRes(
                status=Status(code=Code.OK, message="No model yet"),
                parameters=Parameters(tensor_type="numpy.ndarray", tensors=[]),
            )
        
        try:
            params = self.parameter_extractor.extract_parameters(self.forecaster)
            return GetParametersRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(params),
            )
        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            return GetParametersRes(
                status=Status(code=Code.GET_PARAMETERS_NOT_IMPLEMENTED, message=str(e)),
                parameters=Parameters(tensor_type="numpy.ndarray", tensors=[]),
            )
    
    def fit(self, ins: FitIns) -> FitRes:
        """
        Train model on local data after receiving global parameters.
        
        This is the core of federated learning:
        1. Receive aggregated parameters from server
        2. Initialize local model with these parameters
        3. Train for N local epochs on hospital data
        4. Return updated parameters
        
        Args:
            ins: Instructions from server including global parameters
        
        Returns:
            Updated parameters and training metrics
        """
        logger.info(f"Starting FL training round for {self.hospital_id}")
        
        if self.train_data is None or self.train_data.empty:
            return FitRes(
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message="No training data"),
                parameters=Parameters(tensor_type="numpy.ndarray", tensors=[]),
                num_examples=0,
                metrics={},
            )
        
        try:
            # Initialize forecaster
            self.forecaster = DemandForecaster()
            
            # Apply global parameters if provided
            global_params = parameters_to_ndarrays(ins.parameters)
            if global_params:
                try:
                    self.forecaster = self.parameter_extractor.apply_parameters(
                        self.forecaster, global_params
                    )
                except Exception as e:
                    logger.warning(f"Could not apply global params, training from scratch: {e}")
            
            # Train on local data
            # Prophet trains in one shot, so "local epochs" means retraining
            for epoch in range(settings.fl_local_epochs):
                self.forecaster.fit(self.train_data)
                logger.debug(f"Completed local epoch {epoch + 1}/{settings.fl_local_epochs}")
            
            # Extract updated parameters
            updated_params = self.parameter_extractor.extract_parameters(self.forecaster)
            
            # Compute local metrics
            local_metrics: Dict[str, Scalar] = {}
            if self.test_data is not None and not self.test_data.empty:
                eval_metrics = self.forecaster.evaluate(self.test_data)
                local_metrics = {
                    "mae": float(eval_metrics["mae"]),
                    "rmse": float(eval_metrics["rmse"]),
                    "hospital_id": self.hospital_id,
                }
            
            logger.info(
                f"FL training complete for {self.hospital_id}",
                extra=local_metrics,
            )
            
            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(updated_params),
                num_examples=len(self.train_data),
                metrics=local_metrics,
            )
            
        except Exception as e:
            logger.error(f"FL training failed: {e}")
            return FitRes(
                status=Status(code=Code.FIT_NOT_IMPLEMENTED, message=str(e)),
                parameters=Parameters(tensor_type="numpy.ndarray", tensors=[]),
                num_examples=0,
                metrics={"error": str(e)},
            )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate global model on local test data.
        
        This allows the server to assess model quality across
        all hospitals without accessing their data.
        
        Args:
            ins: Instructions including global model parameters
        
        Returns:
            Evaluation metrics on local test set
        """
        logger.info(f"Evaluating global model at {self.hospital_id}")
        
        if self.test_data is None or self.test_data.empty:
            return EvaluateRes(
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message="No test data"),
                loss=float("inf"),
                num_examples=0,
                metrics={},
            )
        
        try:
            # Create forecaster with global parameters
            self.forecaster = DemandForecaster()
            global_params = parameters_to_ndarrays(ins.parameters)
            
            if global_params:
                self.forecaster = self.parameter_extractor.apply_parameters(
                    self.forecaster, global_params
                )
                
                # Need to fit first if model wasn't properly deserialized
                if not self.forecaster.is_fitted and self.train_data is not None:
                    self.forecaster.fit(self.train_data)
            else:
                # No global params, train locally
                if self.train_data is not None:
                    self.forecaster.fit(self.train_data)
            
            # Evaluate on local test data
            metrics = self.forecaster.evaluate(self.test_data)
            
            logger.info(
                f"Evaluation complete at {self.hospital_id}",
                extra=metrics,
            )
            
            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=float(metrics["mae"]),  # Use MAE as loss
                num_examples=len(self.test_data),
                metrics={
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "mape": float(metrics["mape"]),
                    "coverage": float(metrics["coverage_80pct"]),
                    "hospital_id": self.hospital_id,
                },
            )
            
        except Exception as e:
            logger.error(f"FL evaluation failed: {e}")
            return EvaluateRes(
                status=Status(code=Code.EVALUATE_NOT_IMPLEMENTED, message=str(e)),
                loss=float("inf"),
                num_examples=0,
                metrics={"error": str(e)},
            )


class DemandForecastNumPyClient(fl.client.NumPyClient):
    """
    Simplified NumPy-based FL client (alternative implementation).
    
    This provides a simpler interface when using NumPy arrays directly,
    which is often easier for custom aggregation strategies.
    """
    
    def __init__(
        self,
        hospital_id: str,
        use_synthetic: bool = False,
    ) -> None:
        """Initialize the NumPy client."""
        self.hospital_id = hospital_id
        self.use_synthetic = use_synthetic
        
        self.forecaster: Optional[DemandForecaster] = None
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        
        self._load_local_data()
    
    def _load_local_data(self) -> None:
        """Load local hospital data."""
        loader = DataLoader()
        
        try:
            if self.use_synthetic:
                df = loader.generate_synthetic_data(n_days=365)
            else:
                df = loader.load_aggregated_volume()
            
            if not df.empty:
                self.train_data, self.test_data = split_train_test(df, test_days=30)
        finally:
            loader.close()
    
    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """Return model parameters as numpy arrays."""
        if self.forecaster is None or not self.forecaster.is_fitted:
            return []
        
        return ProphetParameterExtractor.extract_parameters(self.forecaster)
    
    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """
        Train model and return updated parameters.
        
        Returns:
            Tuple of (parameters, num_examples, metrics)
        """
        self.forecaster = DemandForecaster()
        
        # Apply received parameters if any
        if parameters:
            try:
                self.forecaster = ProphetParameterExtractor.apply_parameters(
                    self.forecaster, parameters
                )
            except Exception:
                pass
        
        # Train on local data
        if self.train_data is not None and not self.train_data.empty:
            self.forecaster.fit(self.train_data)
            
            new_params = ProphetParameterExtractor.extract_parameters(self.forecaster)
            
            metrics: Dict[str, Scalar] = {}
            if self.test_data is not None:
                eval_result = self.forecaster.evaluate(self.test_data)
                metrics = {k: float(v) for k, v in eval_result.items()}
            
            return new_params, len(self.train_data), metrics
        
        return [], 0, {}
    
    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """
        Evaluate model on local test data.
        
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        if self.test_data is None or self.test_data.empty:
            return float("inf"), 0, {}
        
        self.forecaster = DemandForecaster()
        
        if parameters:
            self.forecaster = ProphetParameterExtractor.apply_parameters(
                self.forecaster, parameters
            )
        
        # Fit if needed
        if not self.forecaster.is_fitted and self.train_data is not None:
            self.forecaster.fit(self.train_data)
        
        metrics = self.forecaster.evaluate(self.test_data)
        
        return (
            float(metrics["mae"]),
            len(self.test_data),
            {k: float(v) for k, v in metrics.items()},
        )


def start_fl_client(
    server_address: Optional[str] = None,
    hospital_id: Optional[str] = None,
    use_synthetic: bool = False,
    use_numpy_client: bool = True,
) -> None:
    """
    Start the Federated Learning client.
    
    This connects to the FL server and participates in training rounds.
    The client will wait for instructions from the server.
    
    Args:
        server_address: FL server address (default from settings)
        hospital_id: Unique identifier for this hospital
        use_synthetic: Use synthetic data for testing
        use_numpy_client: Use simplified NumPy client interface
    """
    server_address = server_address or settings.fl_server_address
    hospital_id = hospital_id or settings.service_name
    
    logger.info(
        f"Starting FL client",
        extra={
            "server": server_address,
            "hospital_id": hospital_id,
            "use_synthetic": use_synthetic,
        }
    )
    
    if use_numpy_client:
        client = DemandForecastNumPyClient(
            hospital_id=hospital_id,
            use_synthetic=use_synthetic,
        )
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
        )
    else:
        client = DemandForecastClient(
            hospital_id=hospital_id,
            use_synthetic=use_synthetic,
        )
        fl.client.start_client(
            server_address=server_address,
            client=client,
        )


if __name__ == "__main__":
    """
    Command-line interface for starting the FL client.
    
    Usage:
        python fl_client.py [--server ADDRESS] [--hospital-id ID] [--synthetic]
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Start FL client for demand forecasting")
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help=f"FL server address (default: {settings.fl_server_address})",
    )
    parser.add_argument(
        "--hospital-id",
        type=str,
        default=os.environ.get("HOSPITAL_ID", "hospital_default"),
        help="Unique hospital identifier",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--legacy-client",
        action="store_true",
        help="Use legacy Client interface instead of NumPyClient",
    )
    
    args = parser.parse_args()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          DEMAND FORECAST - FEDERATED LEARNING CLIENT         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Hospital ID: {args.hospital_id:<44} ║
    ║  FL Server:   {args.server or settings.fl_server_address:<44} ║
    ║  Data Mode:   {'Synthetic' if args.synthetic else 'Production':<44} ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    start_fl_client(
        server_address=args.server,
        hospital_id=args.hospital_id,
        use_synthetic=args.synthetic,
        use_numpy_client=not args.legacy_client,
    )
