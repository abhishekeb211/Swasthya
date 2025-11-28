"""
Triage & Acuity Agent - XGBoost Classification Model

This module implements the TriageClassifier, an XGBoost-based model for
predicting Emergency Severity Index (ESI) acuity levels from patient data.

================================================================================
WHY XGBOOST FOR TRIAGE CLASSIFICATION?
================================================================================

XGBoost (eXtreme Gradient Boosting) is ideal for triage classification:

1. HANDLES MIXED FEATURE TYPES:
   ─────────────────────────────
   Triage data includes:
   - Continuous: vital signs (HR, BP, SpO2, temp)
   - Categorical: chief complaint category, arrival mode
   - Binary: Red Flag indicators, gender
   XGBoost handles all without extensive preprocessing.

2. HANDLES CLASS IMBALANCE:
   ─────────────────────────
   Acuity levels are not equally distributed:
   - Level 1 (Critical): ~2% of patients
   - Level 2 (Emergent): ~15%
   - Level 3 (Urgent): ~45%
   - Level 4 (Less Urgent): ~30%
   - Level 5 (Non-urgent): ~8%
   
   XGBoost's scale_pos_weight and class weights handle this well.

3. INTERPRETABILITY:
   ──────────────────
   Feature importance helps clinicians trust the model.
   SHAP values can explain individual predictions.

4. FAST INFERENCE:
   ────────────────
   Real-time triage requires <100ms predictions.
   XGBoost's optimized inference is production-ready.

5. FEDERATED LEARNING COMPATIBILITY:
   ──────────────────────────────────
   Tree ensemble parameters (node splits, leaf values) can be
   aggregated across hospital sites for FL training.

================================================================================
FEATURE ENGINEERING FOR TRIAGE
================================================================================

Input features for the model:

VITAL SIGNS (Continuous):
- heart_rate: Heart rate in bpm
- systolic_bp: Systolic blood pressure in mmHg
- diastolic_bp: Diastolic blood pressure in mmHg
- respiratory_rate: Breaths per minute
- oxygen_saturation: SpO2 percentage
- temperature: Body temperature in Celsius
- gcs: Glasgow Coma Scale score (3-15)

PATIENT DEMOGRAPHICS:
- age: Patient age in years
- gender: Binary (0=Female, 1=Male)

COMPLAINT FEATURES:
- chief_complaint_category: Encoded complaint category (0-20)
- pain_score: Self-reported pain (0-10)
- arrival_mode: 0=Walk-in, 1=Ambulance, 2=Helicopter

DERIVED FEATURES:
- shock_index: HR / Systolic BP (elevated = shock)
- map: Mean Arterial Pressure
- pulse_pressure: Systolic - Diastolic
- red_flag_count: Number of Red Flags detected

================================================================================
"""

from __future__ import annotations

import logging
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import settings, AcuityLevel


# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Core features expected by the model
VITAL_SIGN_FEATURES = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "respiratory_rate",
    "oxygen_saturation",
    "temperature",
    "gcs",
]

DEMOGRAPHIC_FEATURES = [
    "age",
    "gender",  # 0=Female, 1=Male
]

COMPLAINT_FEATURES = [
    "chief_complaint_category",
    "pain_score",
    "arrival_mode",
]

DERIVED_FEATURES = [
    "shock_index",
    "map",  # Mean Arterial Pressure
    "pulse_pressure",
    "red_flag_count",
]

ALL_FEATURES = (
    VITAL_SIGN_FEATURES +
    DEMOGRAPHIC_FEATURES +
    COMPLAINT_FEATURES +
    DERIVED_FEATURES
)


# =============================================================================
# TRIAGE CLASSIFIER
# =============================================================================

class TriageClassifier:
    """
    XGBoost-based classifier for Emergency Severity Index (ESI) acuity levels.
    
    This classifier predicts acuity levels 1-5 based on patient vital signs,
    demographics, and complaint features.
    
    Attributes:
        model: The underlying XGBoost classifier
        is_fitted: Whether the model has been trained
        feature_names: List of expected feature names
        class_weights: Weights for handling class imbalance
    
    Example:
        >>> classifier = TriageClassifier()
        >>> classifier.fit(X_train, y_train)
        >>> predictions = classifier.predict(X_test)
        >>> probabilities = classifier.predict_proba(X_test)
    """
    
    # Chief complaint category encoding
    COMPLAINT_CATEGORIES = {
        "chest_pain": 0,
        "abdominal_pain": 1,
        "shortness_of_breath": 2,
        "headache": 3,
        "back_pain": 4,
        "extremity_injury": 5,
        "laceration": 6,
        "fever": 7,
        "nausea_vomiting": 8,
        "dizziness": 9,
        "weakness": 10,
        "altered_mental_status": 11,
        "syncope": 12,
        "fall": 13,
        "motor_vehicle_accident": 14,
        "psychiatric": 15,
        "allergic_reaction": 16,
        "eye_problem": 17,
        "ear_problem": 18,
        "skin_rash": 19,
        "other": 20,
    }
    
    # Arrival mode encoding
    ARRIVAL_MODES = {
        "walk_in": 0,
        "ambulance": 1,
        "helicopter": 2,
    }
    
    def __init__(
        self,
        n_estimators: Optional[int] = None,
        max_depth: Optional[int] = None,
        learning_rate: Optional[float] = None,
        min_child_weight: Optional[int] = None,
        subsample: Optional[float] = None,
        colsample_bytree: Optional[float] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the TriageClassifier.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage (eta)
            min_child_weight: Minimum sum of instance weight in child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns per tree
            random_state: Random seed for reproducibility
        """
        # Use settings defaults if not specified
        self.n_estimators = n_estimators or settings.xgboost_n_estimators
        self.max_depth = max_depth or settings.xgboost_max_depth
        self.learning_rate = learning_rate or settings.xgboost_learning_rate
        self.min_child_weight = min_child_weight or settings.xgboost_min_child_weight
        self.subsample = subsample or settings.xgboost_subsample
        self.colsample_bytree = colsample_bytree or settings.xgboost_colsample_bytree
        self.random_state = random_state
        
        # Initialize model
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_fitted: bool = False
        self.feature_names: List[str] = ALL_FEATURES
        self.training_metadata: Dict[str, Any] = {}
        
        # Class weights for imbalanced data (will be computed during fit)
        self.class_weights: Optional[Dict[int, float]] = None
        
        self._initialize_model()
        
        logger.info(
            "TriageClassifier initialized",
            extra={
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
            }
        )
    
    def _initialize_model(self) -> None:
        """Create a fresh XGBoost classifier instance."""
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=5,  # ESI levels 1-5
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
        )
        
        self.is_fitted = False
        self.training_metadata = {}
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data.
        
        Uses inverse frequency weighting to give more importance
        to rare classes (Level 1 Critical).
        """
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        
        # Inverse frequency weighting
        weights = {
            int(cls): total / (len(unique) * count)
            for cls, count in zip(unique, counts)
        }
        
        return weights
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and engineer features from raw input.
        
        Args:
            df: Raw input DataFrame
        
        Returns:
            DataFrame with all required features
        """
        result = df.copy()
        
        # Compute derived features if base features exist
        if "heart_rate" in result.columns and "systolic_bp" in result.columns:
            # Shock Index: HR / SBP (normal ~0.5-0.7, elevated = shock)
            result["shock_index"] = (
                result["heart_rate"] / result["systolic_bp"].replace(0, np.nan)
            ).fillna(0.7)
        else:
            result["shock_index"] = 0.7
        
        if "systolic_bp" in result.columns and "diastolic_bp" in result.columns:
            # Mean Arterial Pressure: (SBP + 2*DBP) / 3
            result["map"] = (
                result["systolic_bp"] + 2 * result["diastolic_bp"]
            ) / 3
            
            # Pulse Pressure: SBP - DBP
            result["pulse_pressure"] = (
                result["systolic_bp"] - result["diastolic_bp"]
            )
        else:
            result["map"] = 90.0
            result["pulse_pressure"] = 40.0
        
        # Ensure red_flag_count exists
        if "red_flag_count" not in result.columns:
            result["red_flag_count"] = 0
        
        # Fill missing values with defaults
        defaults = {
            "heart_rate": 80.0,
            "systolic_bp": 120.0,
            "diastolic_bp": 80.0,
            "respiratory_rate": 16.0,
            "oxygen_saturation": 98.0,
            "temperature": 37.0,
            "gcs": 15,
            "age": 45,
            "gender": 0,
            "chief_complaint_category": 20,  # "other"
            "pain_score": 5,
            "arrival_mode": 0,  # walk-in
        }
        
        for col, default in defaults.items():
            if col not in result.columns:
                result[col] = default
            else:
                result[col] = result[col].fillna(default)
        
        # Ensure correct column order
        for feature in self.feature_names:
            if feature not in result.columns:
                result[feature] = 0
        
        return result[self.feature_names]
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[List[Tuple]] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = False,
    ) -> "TriageClassifier":
        """
        Train the XGBoost classifier on triage data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target acuity levels (1-5)
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds without improvement to stop
            verbose: Whether to print training progress
        
        Returns:
            self: For method chaining
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        
        # Convert labels to 0-indexed (XGBoost expects 0-4, not 1-5)
        y = np.array(y)
        if y.min() >= 1:
            y = y - 1
        
        # Compute class weights
        self.class_weights = self._compute_class_weights(y)
        
        # Create sample weights
        sample_weights = np.array([self.class_weights[int(yi)] for yi in y])
        
        logger.info(
            f"Training TriageClassifier on {len(X_prepared)} samples",
            extra={
                "features": len(self.feature_names),
                "class_distribution": dict(zip(*np.unique(y, return_counts=True))),
            }
        )
        
        # Reset model
        self._initialize_model()
        
        # Train
        fit_params = {
            "sample_weight": sample_weights,
            "verbose": verbose,
        }
        
        if eval_set:
            # Prepare eval set
            X_eval, y_eval = eval_set[0]
            if isinstance(X_eval, np.ndarray):
                X_eval = pd.DataFrame(X_eval, columns=self.feature_names[:X_eval.shape[1]])
            X_eval = self._prepare_features(X_eval)
            y_eval = np.array(y_eval)
            if y_eval.min() >= 1:
                y_eval = y_eval - 1
            
            fit_params["eval_set"] = [(X_eval, y_eval)]
            fit_params["early_stopping_rounds"] = early_stopping_rounds
        
        self.model.fit(X_prepared, y, **fit_params)
        self.is_fitted = True
        
        # Store training metadata
        self.training_metadata = {
            "n_samples": len(X_prepared),
            "n_features": len(self.feature_names),
            "class_weights": self.class_weights,
            "trained_at": datetime.utcnow().isoformat(),
            "best_iteration": getattr(self.model, "best_iteration", self.n_estimators),
        }
        
        logger.info("TriageClassifier training complete", extra=self.training_metadata)
        
        return self
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict acuity levels for input samples.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of predicted acuity levels (1-5)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        
        # Predict (model returns 0-4, convert to 1-5)
        predictions = self.model.predict(X_prepared)
        
        return predictions + 1
    
    def predict_proba(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        Args:
            X: Feature matrix
        
        Returns:
            Array of shape (n_samples, 5) with probabilities for each acuity level
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
        
        # Prepare features
        X_prepared = self._prepare_features(X)
        
        return self.model.predict_proba(X_prepared)
    
    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, np.ndarray],
    ) -> List[Dict[str, Any]]:
        """
        Predict acuity levels with confidence scores and explanations.
        
        Args:
            X: Feature matrix
        
        Returns:
            List of dictionaries with prediction details
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            # Get top 2 predictions
            sorted_indices = np.argsort(probs)[::-1]
            top_acuity = sorted_indices[0] + 1
            top_prob = probs[sorted_indices[0]]
            second_acuity = sorted_indices[1] + 1
            second_prob = probs[sorted_indices[1]]
            
            results.append({
                "predicted_acuity": int(pred),
                "confidence": float(top_prob),
                "acuity_label": AcuityLevel.get_label(int(pred)),
                "target_time_minutes": AcuityLevel.get_target_time(int(pred)),
                "probabilities": {
                    f"level_{j+1}": float(p)
                    for j, p in enumerate(probs)
                },
                "alternative": {
                    "acuity": int(second_acuity),
                    "probability": float(second_prob),
                },
            })
        
        return results
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X: Feature matrix
            y: True acuity levels
        
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        predictions = self.predict(X)
        y = np.array(y)
        
        # Compute metrics
        accuracy = accuracy_score(y, predictions)
        
        # Per-class metrics
        precision = precision_score(y, predictions, average=None, zero_division=0)
        recall = recall_score(y, predictions, average=None, zero_division=0)
        f1 = f1_score(y, predictions, average=None, zero_division=0)
        
        # Macro and weighted averages
        macro_f1 = f1_score(y, predictions, average="macro", zero_division=0)
        weighted_f1 = f1_score(y, predictions, average="weighted", zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions, labels=[1, 2, 3, 4, 5])
        
        # Critical metric: Under-triage rate for Level 1-2
        # (True critical patients predicted as non-critical)
        critical_mask = y <= 2
        if critical_mask.sum() > 0:
            critical_predictions = predictions[critical_mask]
            under_triage_rate = (critical_predictions > 2).mean()
        else:
            under_triage_rate = 0.0
        
        metrics = {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "under_triage_rate": float(under_triage_rate),
            "per_class": {
                f"level_{i+1}": {
                    "precision": float(precision[i]) if i < len(precision) else 0,
                    "recall": float(recall[i]) if i < len(recall) else 0,
                    "f1": float(f1[i]) if i < len(f1) else 0,
                }
                for i in range(5)
            },
            "confusion_matrix": cm.tolist(),
            "n_samples": len(y),
        }
        
        logger.info(
            "Model evaluation complete",
            extra={
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "under_triage_rate": metrics["under_triage_rate"],
            }
        )
        
        return metrics
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        importance = self.model.feature_importances_
        
        return {
            feature: float(imp)
            for feature, imp in sorted(
                zip(self.feature_names, importance),
                key=lambda x: -x[1]
            )
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "random_state": self.random_state,
        }
    
    def set_params(self, **params) -> "TriageClassifier":
        """Set model hyperparameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self._initialize_model()
        return self
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            path: File path (will save as .json for XGBoost model)
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = path.with_suffix(".json")
        self.model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            "params": self.get_params(),
            "feature_names": self.feature_names,
            "class_weights": self.class_weights,
            "training_metadata": self.training_metadata,
        }
        
        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "TriageClassifier":
        """
        Load a model from disk.
        
        Args:
            path: File path
        
        Returns:
            Loaded TriageClassifier instance
        """
        path = Path(path)
        
        # Load metadata
        metadata_path = path.with_suffix(".meta.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Create instance with saved params
        instance = cls(**metadata["params"])
        instance.feature_names = metadata["feature_names"]
        instance.class_weights = metadata.get("class_weights")
        instance.training_metadata = metadata.get("training_metadata", {})
        
        # Load XGBoost model
        model_path = path.with_suffix(".json")
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(model_path))
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {model_path}")
        
        return instance
    
    def get_booster_params(self) -> List[np.ndarray]:
        """
        Extract booster parameters for federated learning.
        
        Returns XGBoost model parameters as numpy arrays that can be
        aggregated across FL clients.
        """
        if not self.is_fitted:
            return []
        
        # Get the booster
        booster = self.model.get_booster()
        
        # Dump model to JSON and convert to array
        model_json = booster.save_raw("json")
        params_array = np.frombuffer(model_json, dtype=np.uint8)
        
        return [params_array]
    
    def set_booster_params(self, params: List[np.ndarray]) -> None:
        """
        Apply aggregated booster parameters from federated learning.
        
        Args:
            params: List containing serialized model parameters
        """
        if not params:
            return
        
        # Reconstruct model from serialized bytes
        model_bytes = params[0].tobytes()
        
        # Create new booster and load
        booster = xgb.Booster()
        booster.load_model(bytearray(model_bytes))
        
        # Update the classifier's model
        self.model._Booster = booster
        self.is_fitted = True


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_sample_data(n_samples: int = 1000, random_state: int = 42) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic triage data for testing.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed
    
    Returns:
        Tuple of (features DataFrame, labels array)
    """
    np.random.seed(random_state)
    
    # Generate features with realistic distributions
    data = {
        "heart_rate": np.random.normal(80, 20, n_samples).clip(40, 180),
        "systolic_bp": np.random.normal(120, 20, n_samples).clip(70, 200),
        "diastolic_bp": np.random.normal(80, 15, n_samples).clip(40, 120),
        "respiratory_rate": np.random.normal(16, 4, n_samples).clip(8, 40),
        "oxygen_saturation": np.random.normal(97, 3, n_samples).clip(70, 100),
        "temperature": np.random.normal(37.0, 0.8, n_samples).clip(35, 41),
        "gcs": np.random.choice([15, 15, 15, 14, 13, 12, 10, 8, 6], n_samples),
        "age": np.random.normal(50, 20, n_samples).clip(0, 100),
        "gender": np.random.choice([0, 1], n_samples),
        "chief_complaint_category": np.random.randint(0, 21, n_samples),
        "pain_score": np.random.randint(0, 11, n_samples),
        "arrival_mode": np.random.choice([0, 0, 0, 1, 2], n_samples),
        "red_flag_count": np.random.choice([0, 0, 0, 0, 1, 2], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels based on features (simplified logic)
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Critical (Level 1) - ~5%
        if (df.iloc[i]["gcs"] < 9 or
            df.iloc[i]["oxygen_saturation"] < 85 or
            df.iloc[i]["systolic_bp"] < 80 or
            df.iloc[i]["red_flag_count"] >= 2):
            labels[i] = 1
        # Emergent (Level 2) - ~15%
        elif (df.iloc[i]["gcs"] < 13 or
              df.iloc[i]["oxygen_saturation"] < 92 or
              df.iloc[i]["heart_rate"] > 130 or
              df.iloc[i]["arrival_mode"] == 2 or
              df.iloc[i]["red_flag_count"] == 1):
            labels[i] = 2
        # Urgent (Level 3) - ~40%
        elif (df.iloc[i]["pain_score"] >= 7 or
              df.iloc[i]["arrival_mode"] == 1 or
              df.iloc[i]["chief_complaint_category"] in [0, 1, 2]):
            labels[i] = 3
        # Less Urgent (Level 4) - ~30%
        elif df.iloc[i]["pain_score"] >= 4:
            labels[i] = 4
        # Non-urgent (Level 5) - ~10%
        else:
            labels[i] = 5
    
    return df, labels
