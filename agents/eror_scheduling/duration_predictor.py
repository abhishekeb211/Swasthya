"""
Surgery Duration Predictor Module.

Uses XGBoost to predict surgery duration based on:
- Procedure type
- Patient acuity level
- Additional factors (age, BMI, comorbidities)

This module demonstrates ML integration within a hybrid optimization agent.
The predictions feed directly into the OR-Tools scheduler.
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from config import config, PROCEDURE_TYPES

logger = logging.getLogger(__name__)


@dataclass
class PredictionInput:
    """Input features for duration prediction."""
    procedure_type: str
    patient_acuity: int  # ESI 1-5
    patient_age: int = 50
    patient_bmi: float = 25.0
    comorbidity_count: int = 0
    surgeon_experience_years: int = 10
    is_emergency: bool = False


@dataclass
class PredictionResult:
    """Output from duration prediction."""
    predicted_duration_minutes: int
    confidence_interval_lower: int
    confidence_interval_upper: int
    feature_importance: Dict[str, float]


class DurationPredictor:
    """
    XGBoost-based surgery duration predictor.
    
    This predictor uses gradient boosting to estimate surgery duration,
    which is then used as input to the OR scheduling optimization.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """Initialize the duration predictor."""
        self.model_path = model_path or config.model.model_path
        self.model: Optional[xgb.Booster] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = [
            "procedure_type_encoded",
            "procedure_complexity",
            "procedure_base_duration",
            "patient_acuity",
            "patient_age",
            "patient_bmi",
            "comorbidity_count",
            "surgeon_experience_years",
            "is_emergency"
        ]
        self._initialize_encoders()
    
    def _initialize_encoders(self) -> None:
        """Initialize label encoders for categorical features."""
        procedure_encoder = LabelEncoder()
        procedure_encoder.fit(list(PROCEDURE_TYPES.keys()))
        self.label_encoders["procedure_type"] = procedure_encoder
    
    def _prepare_features(self, input_data: PredictionInput) -> np.ndarray:
        """Convert input to feature vector."""
        procedure_info = PROCEDURE_TYPES.get(
            input_data.procedure_type,
            {"base_duration": 90, "complexity": 2}
        )
        
        # Encode procedure type
        try:
            proc_encoded = self.label_encoders["procedure_type"].transform(
                [input_data.procedure_type]
            )[0]
        except ValueError:
            proc_encoded = 0  # Default for unknown procedures
        
        features = np.array([
            proc_encoded,
            procedure_info["complexity"],
            procedure_info["base_duration"],
            input_data.patient_acuity,
            input_data.patient_age,
            input_data.patient_bmi,
            input_data.comorbidity_count,
            input_data.surgeon_experience_years,
            1 if input_data.is_emergency else 0
        ]).reshape(1, -1)
        
        return features
    
    def load_model(self) -> bool:
        """Load the XGBoost model from disk."""
        try:
            if self.model_path.exists():
                self.model = xgb.Booster()
                self.model.load_model(str(self.model_path))
                logger.info(f"Loaded duration predictor model from {self.model_path}")
                return True
            else:
                logger.warning(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self) -> bool:
        """Save the XGBoost model to disk."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(self.model_path))
            logger.info(f"Saved duration predictor model to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """
        Predict surgery duration for given input.
        
        Returns predicted duration with confidence intervals.
        """
        if self.model is None:
            # Fallback to heuristic if model not loaded
            return self._heuristic_prediction(input_data)
        
        features = self._prepare_features(input_data)
        dmatrix = xgb.DMatrix(features, feature_names=self.feature_names)
        
        # Get prediction
        prediction = float(self.model.predict(dmatrix)[0])
        
        # Calculate confidence interval (±15% for demonstration)
        margin = prediction * 0.15
        
        # Get feature importance
        importance = self.model.get_score(importance_type='gain')
        total_gain = sum(importance.values()) if importance else 1
        normalized_importance = {
            k: v / total_gain for k, v in importance.items()
        } if importance else {}
        
        return PredictionResult(
            predicted_duration_minutes=int(max(30, prediction)),
            confidence_interval_lower=int(max(30, prediction - margin)),
            confidence_interval_upper=int(prediction + margin),
            feature_importance=normalized_importance
        )
    
    def _heuristic_prediction(self, input_data: PredictionInput) -> PredictionResult:
        """Fallback heuristic-based prediction when model unavailable."""
        procedure_info = PROCEDURE_TYPES.get(
            input_data.procedure_type,
            {"base_duration": 90, "complexity": 2}
        )
        
        base = procedure_info["base_duration"]
        
        # Adjust for acuity (higher acuity = more complex patient)
        acuity_factor = 1.0 + (5 - input_data.patient_acuity) * 0.05
        
        # Adjust for age
        age_factor = 1.0 + max(0, (input_data.patient_age - 60)) * 0.005
        
        # Adjust for BMI
        bmi_factor = 1.0 + max(0, (input_data.patient_bmi - 30)) * 0.01
        
        # Adjust for comorbidities
        comorbidity_factor = 1.0 + input_data.comorbidity_count * 0.05
        
        # Adjust for emergency status
        emergency_factor = 1.2 if input_data.is_emergency else 1.0
        
        # Calculate final prediction
        prediction = base * acuity_factor * age_factor * bmi_factor * comorbidity_factor * emergency_factor
        margin = prediction * 0.15
        
        return PredictionResult(
            predicted_duration_minutes=int(prediction),
            confidence_interval_lower=int(prediction - margin),
            confidence_interval_upper=int(prediction + margin),
            feature_importance={"heuristic": 1.0}
        )
    
    def predict_batch(self, inputs: List[PredictionInput]) -> List[PredictionResult]:
        """Predict durations for multiple surgeries."""
        return [self.predict(inp) for inp in inputs]


def generate_training_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for the duration predictor.
    
    In production, this would be replaced with historical surgery data.
    """
    np.random.seed(42)
    
    procedure_types = list(PROCEDURE_TYPES.keys())
    procedure_encoder = LabelEncoder()
    procedure_encoder.fit(procedure_types)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random procedure
        proc_type = np.random.choice(procedure_types)
        proc_info = PROCEDURE_TYPES[proc_type]
        
        # Random patient characteristics
        acuity = np.random.randint(1, 6)
        age = np.random.randint(18, 90)
        bmi = np.random.uniform(18, 45)
        comorbidities = np.random.randint(0, 6)
        surgeon_exp = np.random.randint(1, 30)
        is_emergency = np.random.random() < 0.2
        
        # Feature vector
        features = [
            procedure_encoder.transform([proc_type])[0],
            proc_info["complexity"],
            proc_info["base_duration"],
            acuity,
            age,
            bmi,
            comorbidities,
            surgeon_exp,
            1 if is_emergency else 0
        ]
        
        # Generate target with realistic variations
        base_duration = proc_info["base_duration"]
        
        # Apply factors
        duration = base_duration
        duration *= (1 + (5 - acuity) * 0.05)  # Higher acuity = more complex
        duration *= (1 + max(0, age - 60) * 0.005)  # Older patients take longer
        duration *= (1 + max(0, bmi - 30) * 0.01)  # Higher BMI takes longer
        duration *= (1 + comorbidities * 0.05)  # More comorbidities take longer
        duration *= (1 - min(surgeon_exp, 20) * 0.01)  # Experience reduces time
        duration *= 1.2 if is_emergency else 1.0  # Emergencies take longer
        
        # Add noise
        duration += np.random.normal(0, duration * 0.1)
        duration = max(30, duration)  # Minimum 30 minutes
        
        X.append(features)
        y.append(duration)
    
    return np.array(X), np.array(y)


def train_dummy_model(model_path: Optional[Path] = None) -> DurationPredictor:
    """
    Train a dummy XGBoost model for surgery duration prediction.
    
    This creates a functional model using synthetic data.
    In production, replace with actual historical surgery data.
    """
    model_path = model_path or config.model.model_path
    
    logger.info("Generating synthetic training data...")
    X, y = generate_training_data(n_samples=2000)
    
    feature_names = [
        "procedure_type_encoded",
        "procedure_complexity",
        "procedure_base_duration",
        "patient_acuity",
        "patient_age",
        "patient_bmi",
        "comorbidity_count",
        "surgeon_experience_years",
        "is_emergency"
    ]
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    # Train model
    logger.info("Training XGBoost model...")
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evallist,
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Create and return predictor with loaded model
    predictor = DurationPredictor(model_path)
    predictor.model = model
    
    # Log model performance
    val_predictions = model.predict(dval)
    rmse = np.sqrt(np.mean((val_predictions - y_val) ** 2))
    logger.info(f"Model validation RMSE: {rmse:.2f} minutes")
    
    return predictor


def get_or_create_predictor() -> DurationPredictor:
    """
    Get existing predictor or train a new one.
    
    This is called on application startup.
    """
    predictor = DurationPredictor()
    
    if predictor.load_model():
        logger.info("Loaded existing duration prediction model")
        return predictor
    
    if config.model.retrain_on_startup:
        logger.info("Training new duration prediction model...")
        return train_dummy_model()
    
    logger.warning("Using heuristic-based predictions (no model loaded)")
    return predictor
