"""
Demand Forecast Agent - Core Forecasting Model

This module implements the DemandForecaster class, which wraps Facebook Prophet
for hospital patient volume prediction.

================================================================================
ARCHITECTURAL DECISION: Why Prophet over ARIMA for Hospital Demand Forecasting?
================================================================================

Facebook Prophet is chosen as the primary forecasting engine for several 
domain-specific reasons:

1. MULTIPLE SEASONALITIES (Critical for Healthcare):
   ─────────────────────────────────────────────────
   Hospital data exhibits complex overlapping seasonal patterns:
   - Daily: ER visits peak 6-10 PM, admissions peak mid-morning
   - Weekly: Weekends see 20-30% more ER visits, elective admissions drop
   - Yearly: Flu season (Nov-Mar), summer trauma, post-holiday lull
   
   Prophet handles additive/multiplicative seasonalities natively.
   ARIMA requires manual Fourier terms or SARIMA with single seasonality.

2. HOLIDAY EFFECTS (Hospital-Specific):
   ─────────────────────────────────────
   Major holidays dramatically affect patient volume:
   - Christmas/Thanksgiving: Delayed elective procedures, ER spikes
   - July 4th/Labor Day: Trauma increases
   - Flu vaccination campaigns affect respiratory admissions
   
   Prophet's built-in holiday modeling is far simpler than ARIMA regressors.

3. ROBUSTNESS TO MISSING DATA:
   ────────────────────────────
   Healthcare data often has gaps (system outages, data migration).
   Prophet handles missing values gracefully; ARIMA requires imputation.

4. INTERPRETABILITY FOR STAKEHOLDERS:
   ───────────────────────────────────
   Hospital administrators need to understand forecasts.
   Prophet's decomposition (trend + seasonality + holidays) is intuitive.
   ARIMA's (p,d,q) parameters are opaque to non-statisticians.

5. CHANGEPOINT DETECTION:
   ───────────────────────
   Hospital demand shifts (new facility, policy changes, pandemics).
   Prophet automatically detects trend changepoints.
   ARIMA requires manual intervention selection.

WHEN TO CONSIDER ARIMA INSTEAD:
───────────────────────────────
- Very short time series (< 2 years of data)
- Strictly stationary data with single seasonality
- Real-time streaming predictions (Prophet is batch-oriented)
- Memory-constrained environments (Prophet's Stan backend is heavier)

================================================================================
"""

from __future__ import annotations

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

from config import settings

# Configure module logger
logger = logging.getLogger(__name__)


class DemandForecaster:
    """
    A wrapper class for Facebook Prophet tailored to hospital demand forecasting.
    
    This class encapsulates:
    - Model initialization with healthcare-appropriate defaults
    - Custom holiday calendars for hospital operations
    - Training and prediction pipelines
    - Serialization for MLflow and Federated Learning
    
    Attributes:
        model: The underlying Prophet model instance
        is_fitted: Whether the model has been trained
        training_metadata: Information about the last training run
    
    Example:
        >>> forecaster = DemandForecaster()
        >>> forecaster.fit(historical_data)
        >>> predictions = forecaster.predict(horizon_hours=168)
    """
    
    # Hospital-relevant US holidays
    HOSPITAL_HOLIDAYS: List[Dict[str, Any]] = [
        {"holiday": "new_years", "ds": "2024-01-01", "lower_window": -1, "upper_window": 1},
        {"holiday": "mlk_day", "ds": "2024-01-15", "lower_window": 0, "upper_window": 0},
        {"holiday": "presidents_day", "ds": "2024-02-19", "lower_window": 0, "upper_window": 0},
        {"holiday": "memorial_day", "ds": "2024-05-27", "lower_window": -1, "upper_window": 1},
        {"holiday": "independence_day", "ds": "2024-07-04", "lower_window": -1, "upper_window": 1},
        {"holiday": "labor_day", "ds": "2024-09-02", "lower_window": -1, "upper_window": 1},
        {"holiday": "thanksgiving", "ds": "2024-11-28", "lower_window": -1, "upper_window": 1},
        {"holiday": "christmas", "ds": "2024-12-25", "lower_window": -2, "upper_window": 1},
        # Flu season markers (custom hospital events)
        {"holiday": "flu_season_start", "ds": "2024-11-01", "lower_window": 0, "upper_window": 14},
        {"holiday": "flu_season_peak", "ds": "2025-01-15", "lower_window": -7, "upper_window": 7},
    ]
    
    def __init__(
        self,
        yearly_seasonality: bool = None,
        weekly_seasonality: bool = None,
        daily_seasonality: bool = None,
        changepoint_prior_scale: float = None,
        seasonality_prior_scale: float = None,
        holidays_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the DemandForecaster with healthcare-optimized defaults.
        
        Args:
            yearly_seasonality: Enable yearly patterns (flu season, summer trauma)
            weekly_seasonality: Enable weekly patterns (weekend ER spikes)
            daily_seasonality: Enable hourly patterns (evening ER peaks)
            changepoint_prior_scale: Trend flexibility (0.001-0.5)
            seasonality_prior_scale: Seasonality strength (0.01-100)
            holidays_df: Custom holiday DataFrame (uses HOSPITAL_HOLIDAYS if None)
        """
        # Use settings defaults if not specified
        self.yearly_seasonality = yearly_seasonality or settings.prophet_yearly_seasonality
        self.weekly_seasonality = weekly_seasonality or settings.prophet_weekly_seasonality
        self.daily_seasonality = daily_seasonality or settings.prophet_daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale or settings.prophet_changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale or settings.prophet_seasonality_prior_scale
        
        # Prepare holidays DataFrame
        if holidays_df is not None:
            self.holidays_df = holidays_df
        else:
            self.holidays_df = self._build_hospital_holidays()
        
        # Initialize Prophet model
        self.model: Optional[Prophet] = None
        self.is_fitted: bool = False
        self.training_metadata: Dict[str, Any] = {}
        
        self._initialize_model()
        
        logger.info(
            "DemandForecaster initialized",
            extra={
                "yearly_seasonality": self.yearly_seasonality,
                "weekly_seasonality": self.weekly_seasonality,
                "daily_seasonality": self.daily_seasonality,
            }
        )
    
    def _build_hospital_holidays(self) -> pd.DataFrame:
        """
        Build a multi-year holiday DataFrame for Prophet.
        
        Extends the base HOSPITAL_HOLIDAYS template across multiple years
        to ensure forecasts handle future holidays correctly.
        
        Returns:
            DataFrame with columns: holiday, ds, lower_window, upper_window
        """
        all_holidays = []
        base_year = 2024
        
        for year_offset in range(-2, 5):  # 2022-2028
            for holiday in self.HOSPITAL_HOLIDAYS:
                try:
                    base_date = pd.to_datetime(holiday["ds"])
                    adjusted_date = base_date.replace(year=base_year + year_offset)
                    all_holidays.append({
                        "holiday": holiday["holiday"],
                        "ds": adjusted_date,
                        "lower_window": holiday["lower_window"],
                        "upper_window": holiday["upper_window"],
                    })
                except ValueError:
                    # Handle leap year edge cases
                    continue
        
        return pd.DataFrame(all_holidays)
    
    def _initialize_model(self) -> None:
        """
        Create a fresh Prophet model instance with configured parameters.
        
        This method is called during initialization and can be used to
        reset the model for retraining.
        """
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays=self.holidays_df,
            uncertainty_samples=1000,  # For confidence intervals
            mcmc_samples=0,  # Use MAP estimation for speed (set >0 for full Bayesian)
        )
        
        # Add custom seasonalities for hospital-specific patterns
        # Shift change effects (8-hour cycles in nursing)
        self.model.add_seasonality(
            name="shift_change",
            period=8/24,  # 8 hours expressed in days
            fourier_order=3,
        )
        
        self.is_fitted = False
        self.training_metadata = {}
    
    def fit(
        self,
        df: pd.DataFrame,
        target_column: str = "patient_count",
        datetime_column: str = "timestamp",
    ) -> "DemandForecaster":
        """
        Train the Prophet model on historical patient volume data.
        
        Args:
            df: Historical data with datetime and target columns
            target_column: Name of the column containing patient counts
            datetime_column: Name of the datetime column
        
        Returns:
            self: For method chaining
        
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        # Validate input
        if df.empty:
            raise ValueError("Cannot fit on empty DataFrame")
        
        required_columns = [datetime_column, target_column]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Prepare data in Prophet's expected format
        prophet_df = df[[datetime_column, target_column]].copy()
        prophet_df.columns = ["ds", "y"]
        
        # Ensure datetime type
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        
        # Remove duplicates (Prophet requires unique timestamps)
        prophet_df = prophet_df.groupby("ds").agg({"y": "sum"}).reset_index()
        
        # Handle negative values (shouldn't happen for counts, but be safe)
        prophet_df["y"] = prophet_df["y"].clip(lower=0)
        
        # Log transform for better handling of multiplicative patterns
        # Add 1 to handle zeros
        prophet_df["y"] = np.log1p(prophet_df["y"])
        
        logger.info(
            f"Fitting Prophet model on {len(prophet_df)} observations",
            extra={
                "date_range": f"{prophet_df['ds'].min()} to {prophet_df['ds'].max()}",
                "mean_value": float(prophet_df['y'].mean()),
            }
        )
        
        # Reset model to ensure clean state
        self._initialize_model()
        
        # Fit the model
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        # Store training metadata for tracking
        self.training_metadata = {
            "training_start": prophet_df["ds"].min().isoformat(),
            "training_end": prophet_df["ds"].max().isoformat(),
            "n_observations": len(prophet_df),
            "mean_value": float(np.expm1(prophet_df["y"]).mean()),
            "trained_at": datetime.utcnow().isoformat(),
        }
        
        logger.info("Prophet model fitted successfully", extra=self.training_metadata)
        
        return self
    
    def predict(
        self,
        horizon_hours: int = None,
        start_date: Optional[datetime] = None,
        include_history: bool = False,
    ) -> pd.DataFrame:
        """
        Generate patient volume forecasts.
        
        Args:
            horizon_hours: Number of hours to forecast (default from settings)
            start_date: Start of forecast period (default: now)
            include_history: Whether to include historical fitted values
        
        Returns:
            DataFrame with columns:
                - ds: Timestamp
                - yhat: Point forecast
                - yhat_lower: Lower confidence bound (80%)
                - yhat_upper: Upper confidence bound (80%)
        
        Raises:
            RuntimeError: If model has not been fitted
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")
        
        horizon_hours = horizon_hours or settings.forecast_horizon_hours
        start_date = start_date or datetime.utcnow()
        
        # Create future DataFrame
        if include_history:
            future = self.model.make_future_dataframe(
                periods=horizon_hours,
                freq="H",
                include_history=True,
            )
        else:
            future = pd.DataFrame({
                "ds": pd.date_range(
                    start=start_date,
                    periods=horizon_hours,
                    freq="H",
                )
            })
        
        # Generate predictions
        forecast = self.model.predict(future)
        
        # Select and rename relevant columns
        result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        
        # Reverse log transform
        result["yhat"] = np.expm1(result["yhat"]).clip(lower=0)
        result["yhat_lower"] = np.expm1(result["yhat_lower"]).clip(lower=0)
        result["yhat_upper"] = np.expm1(result["yhat_upper"]).clip(lower=0)
        
        # Round to whole numbers (can't have fractional patients)
        result["yhat"] = result["yhat"].round().astype(int)
        result["yhat_lower"] = result["yhat_lower"].round().astype(int)
        result["yhat_upper"] = result["yhat_upper"].round().astype(int)
        
        logger.info(
            f"Generated {len(result)} hourly predictions",
            extra={
                "forecast_start": result["ds"].min().isoformat(),
                "forecast_end": result["ds"].max().isoformat(),
                "mean_prediction": float(result["yhat"].mean()),
            }
        )
        
        return result
    
    def evaluate(
        self,
        test_df: pd.DataFrame,
        target_column: str = "patient_count",
        datetime_column: str = "timestamp",
    ) -> Dict[str, float]:
        """
        Evaluate model performance on held-out test data.
        
        Computes standard forecasting metrics:
        - MAE: Mean Absolute Error (interpretable in patient counts)
        - RMSE: Root Mean Square Error (penalizes large errors)
        - MAPE: Mean Absolute Percentage Error (scale-independent)
        - Coverage: % of actuals within prediction interval
        
        Args:
            test_df: Test data with actual values
            target_column: Name of the target column
            datetime_column: Name of the datetime column
        
        Returns:
            Dictionary of metric names to values
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        
        # Prepare test data
        test_prophet = test_df[[datetime_column, target_column]].copy()
        test_prophet.columns = ["ds", "y_actual"]
        test_prophet["ds"] = pd.to_datetime(test_prophet["ds"])
        
        # Generate predictions for test period
        forecast = self.model.predict(test_prophet[["ds"]])
        
        # Merge actuals with predictions
        merged = test_prophet.merge(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]], on="ds")
        
        # Reverse log transform on predictions
        merged["yhat"] = np.expm1(merged["yhat"])
        merged["yhat_lower"] = np.expm1(merged["yhat_lower"])
        merged["yhat_upper"] = np.expm1(merged["yhat_upper"])
        
        # Calculate metrics
        y_actual = merged["y_actual"].values
        y_pred = merged["yhat"].values
        
        mae = float(np.mean(np.abs(y_actual - y_pred)))
        rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))
        
        # MAPE (handle zeros carefully)
        nonzero_mask = y_actual > 0
        if nonzero_mask.sum() > 0:
            mape = float(np.mean(np.abs((y_actual[nonzero_mask] - y_pred[nonzero_mask]) / y_actual[nonzero_mask]))) * 100
        else:
            mape = float("nan")
        
        # Coverage: % of actuals within prediction interval
        coverage = float(np.mean(
            (y_actual >= merged["yhat_lower"]) & (y_actual <= merged["yhat_upper"])
        )) * 100
        
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "coverage_80pct": coverage,
            "n_test_samples": len(merged),
        }
        
        logger.info("Model evaluation complete", extra=metrics)
        
        return metrics
    
    def get_components(self) -> pd.DataFrame:
        """
        Extract trend and seasonality components for interpretability.
        
        Returns:
            DataFrame with decomposed forecast components
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted first")
        
        # Generate forecast with components
        future = self.model.make_future_dataframe(periods=168, freq="H")
        forecast = self.model.predict(future)
        
        component_cols = ["ds", "trend", "weekly", "yearly"]
        if "daily" in forecast.columns:
            component_cols.append("daily")
        if "holidays" in forecast.columns:
            component_cols.append("holidays")
        if "shift_change" in forecast.columns:
            component_cols.append("shift_change")
        
        return forecast[[c for c in component_cols if c in forecast.columns]]
    
    def serialize(self) -> str:
        """
        Serialize the model to JSON string for storage/transfer.
        
        Prophet's native JSON serialization is preferred over pickle
        for better cross-version compatibility.
        
        Returns:
            JSON string representation of the model
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize unfitted model")
        
        return model_to_json(self.model)
    
    @classmethod
    def deserialize(cls, json_str: str) -> "DemandForecaster":
        """
        Reconstruct a DemandForecaster from a serialized JSON string.
        
        Args:
            json_str: JSON string from serialize()
        
        Returns:
            New DemandForecaster instance with loaded model
        """
        instance = cls()
        instance.model = model_from_json(json_str)
        instance.is_fitted = True
        return instance
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to a file.
        
        Args:
            path: File path for saving (should end with .json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            f.write(self.serialize())
        
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "DemandForecaster":
        """
        Load a model from a file.
        
        Args:
            path: Path to saved model file
        
        Returns:
            Loaded DemandForecaster instance
        """
        with open(path, "r") as f:
            json_str = f.read()
        
        return cls.deserialize(json_str)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters for MLflow logging.
        
        Returns:
            Dictionary of parameter names to values
        """
        return {
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
        }
