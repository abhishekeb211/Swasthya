"""
Demand Forecast Agent - MLflow Tracking Module

This module provides utilities for experiment tracking and model versioning
using MLflow. It handles:

1. Experiment creation and management
2. Metric logging (MAE, RMSE, MAPE, Coverage)
3. Parameter logging (Prophet hyperparameters)
4. Model artifact storage (serialized Prophet models)
5. Model registry operations (staging, production deployment)

================================================================================
WHY MLFLOW FOR HOSPITAL AI?
================================================================================

MLflow provides critical capabilities for healthcare ML systems:

1. REPRODUCIBILITY (Regulatory Requirement):
   - Full audit trail of model versions
   - Parameter and metric history
   - Essential for FDA/compliance documentation

2. MODEL LINEAGE:
   - Track which data version trained which model
   - Critical when investigating prediction failures

3. A/B TESTING:
   - Compare model versions side-by-side
   - Gradual rollout of new forecasting models

4. ROLLBACK CAPABILITY:
   - Quick reversion to previous model versions
   - Essential for 24/7 hospital operations

================================================================================
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from config import settings

# Configure module logger
logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    A wrapper class for MLflow operations tailored to the Demand Forecast Agent.
    
    This class provides a simplified interface for common MLflow operations,
    with proper error handling and logging for production use.
    
    Attributes:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        client: MLflow client for advanced operations
    
    Example:
        >>> tracker = MLflowTracker()
        >>> with tracker.start_run(run_name="daily_retrain") as run:
        ...     tracker.log_params({"changepoint_prior_scale": 0.05})
        ...     tracker.log_metrics({"mae": 12.5, "rmse": 15.2})
        ...     tracker.log_model(forecaster, "prophet_model")
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking server URI (default from settings)
            experiment_name: Experiment name (default from settings)
        """
        self.tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        
        # Configure MLflow
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        self._setup_experiment()
        
        # Initialize client for model registry operations
        self.client = MlflowClient(tracking_uri=self.tracking_uri)
        
        logger.info(
            "MLflow tracker initialized",
            extra={
                "tracking_uri": self.tracking_uri,
                "experiment_name": self.experiment_name,
            }
        )
    
    def _setup_experiment(self) -> None:
        """
        Create or retrieve the MLflow experiment.
        
        Sets the experiment as active for subsequent operations.
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    self.experiment_name,
                    tags={
                        "project": "hospital-ai",
                        "agent": "demand-forecast",
                        "model_type": "prophet",
                    }
                )
                logger.info(f"Created new experiment: {self.experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {self.experiment_name} (ID: {experiment_id})")
            
            mlflow.set_experiment(self.experiment_name)
            self.experiment_id = experiment_id
            
        except MlflowException as e:
            logger.error(f"Failed to setup MLflow experiment: {e}")
            raise
    
    @contextmanager
    def start_run(
        self,
        run_name: Optional[str] = None,
        nested: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ) -> Generator[mlflow.ActiveRun, None, None]:
        """
        Context manager for MLflow runs.
        
        Args:
            run_name: Human-readable name for the run
            nested: Whether this is a nested run (for hyperparameter tuning)
            tags: Additional tags for the run
        
        Yields:
            Active MLflow run
        
        Example:
            >>> with tracker.start_run(run_name="experiment_1") as run:
            ...     # logging operations here
            ...     pass
        """
        default_tags = {
            "environment": settings.environment,
            "service": settings.service_name,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        if tags:
            default_tags.update(tags)
        
        run_name = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            with mlflow.start_run(run_name=run_name, nested=nested, tags=default_tags) as run:
                logger.info(f"Started MLflow run: {run.info.run_id}")
                yield run
                logger.info(f"Completed MLflow run: {run.info.run_id}")
                
        except MlflowException as e:
            logger.error(f"MLflow run failed: {e}")
            raise
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters for the current run.
        
        Args:
            params: Dictionary of parameter names to values
        """
        try:
            # MLflow only accepts strings, so convert values
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
            logger.debug(f"Logged {len(params)} parameters")
        except MlflowException as e:
            logger.error(f"Failed to log parameters: {e}")
            raise
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics for the current run.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (for time series of metrics)
        """
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.debug(f"Logged {len(metrics)} metrics: {metrics}")
        except MlflowException as e:
            logger.error(f"Failed to log metrics: {e}")
            raise
    
    def log_model(
        self,
        forecaster: "DemandForecaster",
        artifact_path: str = "prophet_model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log a DemandForecaster model as an artifact.
        
        The model is serialized to JSON and saved as an artifact.
        Optionally registers the model in MLflow Model Registry.
        
        Args:
            forecaster: The trained DemandForecaster instance
            artifact_path: Path within the artifact store
            registered_model_name: If provided, register in Model Registry
        """
        try:
            # Create temporary directory for model artifacts
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = Path(tmpdir) / "model.json"
                
                # Save model to temporary file
                forecaster.save(model_path)
                
                # Log as artifact
                mlflow.log_artifact(str(model_path), artifact_path)
                
                # Log model metadata
                metadata_path = Path(tmpdir) / "metadata.json"
                import json
                with open(metadata_path, "w") as f:
                    json.dump(forecaster.training_metadata, f, indent=2)
                mlflow.log_artifact(str(metadata_path), artifact_path)
                
                logger.info(f"Logged model artifact to {artifact_path}")
            
            # Register model if requested
            if registered_model_name:
                self._register_model(artifact_path, registered_model_name)
                
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
            raise
    
    def _register_model(
        self,
        artifact_path: str,
        model_name: str,
    ) -> ModelVersion:
        """
        Register a logged model in MLflow Model Registry.
        
        Args:
            artifact_path: Path to the model artifact
            model_name: Name for the registered model
        
        Returns:
            ModelVersion object for the registered model
        """
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        
        try:
            # Check if model exists in registry
            try:
                self.client.get_registered_model(model_name)
            except MlflowException:
                # Create new registered model
                self.client.create_registered_model(
                    model_name,
                    tags={
                        "task": "demand_forecasting",
                        "algorithm": "prophet",
                    },
                    description="Hospital demand forecasting model using Facebook Prophet",
                )
                logger.info(f"Created registered model: {model_name}")
            
            # Create new version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
                tags={
                    "environment": settings.environment,
                },
            )
            
            logger.info(
                f"Registered model version: {model_name} v{model_version.version}",
                extra={
                    "model_name": model_name,
                    "version": model_version.version,
                    "source": model_uri,
                }
            )
            
            return model_version
            
        except MlflowException as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def transition_model_stage(
        self,
        model_name: str,
        version: int,
        stage: str,
    ) -> None:
        """
        Transition a model version to a new stage.
        
        Args:
            model_name: Name of the registered model
            version: Version number to transition
            stage: Target stage (Staging, Production, Archived)
        """
        valid_stages = ["Staging", "Production", "Archived", "None"]
        if stage not in valid_stages:
            raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
        
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
            )
            logger.info(f"Transitioned {model_name} v{version} to {stage}")
            
        except MlflowException as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise
    
    def get_latest_model_version(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[ModelVersion]:
        """
        Get the latest model version in a given stage.
        
        Args:
            model_name: Name of the registered model
            stage: Stage to query (default: Production)
        
        Returns:
            ModelVersion or None if no model in stage
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0]
            return None
        except MlflowException as e:
            logger.warning(f"Could not get latest model version: {e}")
            return None
    
    def load_model(
        self,
        model_name: str,
        version: Optional[int] = None,
        stage: str = "Production",
    ) -> "DemandForecaster":
        """
        Load a model from the Model Registry.
        
        Args:
            model_name: Name of the registered model
            version: Specific version to load (optional)
            stage: Stage to load from if version not specified
        
        Returns:
            Loaded DemandForecaster instance
        """
        from model import DemandForecaster
        
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
            else:
                versions = self.client.get_latest_versions(model_name, stages=[stage])
                if not versions:
                    raise ValueError(f"No model found in stage: {stage}")
                model_version = versions[0]
            
            # Download model artifact
            artifact_path = self.client.download_artifacts(
                model_version.run_id,
                "prophet_model/model.json",
            )
            
            # Load and return model
            forecaster = DemandForecaster.load(artifact_path)
            
            logger.info(
                f"Loaded model: {model_name} v{model_version.version}",
                extra={"stage": model_version.current_stage}
            )
            
            return forecaster
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


def log_training_run(
    forecaster: "DemandForecaster",
    metrics: Dict[str, float],
    run_name: Optional[str] = None,
    register_model: bool = True,
) -> str:
    """
    Convenience function to log a complete training run.
    
    This is a high-level function that encapsulates the common pattern
    of logging parameters, metrics, and model artifact in one call.
    
    Args:
        forecaster: Trained DemandForecaster instance
        metrics: Evaluation metrics dictionary
        run_name: Optional name for the run
        register_model: Whether to register in Model Registry
    
    Returns:
        MLflow run ID
    
    Example:
        >>> forecaster = DemandForecaster()
        >>> forecaster.fit(train_data)
        >>> metrics = forecaster.evaluate(test_data)
        >>> run_id = log_training_run(forecaster, metrics)
    """
    tracker = MLflowTracker()
    
    with tracker.start_run(run_name=run_name) as run:
        # Log model parameters
        tracker.log_params(forecaster.get_params())
        
        # Log training metadata
        if forecaster.training_metadata:
            tracker.log_params({
                f"data_{k}": v 
                for k, v in forecaster.training_metadata.items()
            })
        
        # Log evaluation metrics
        tracker.log_metrics(metrics)
        
        # Log model artifact
        registered_name = settings.mlflow_model_registry_name if register_model else None
        tracker.log_model(
            forecaster,
            artifact_path="prophet_model",
            registered_model_name=registered_name,
        )
        
        return run.info.run_id
