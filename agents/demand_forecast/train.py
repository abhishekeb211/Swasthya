"""
Demand Forecast Agent - Training Script

This module handles the end-to-end training pipeline:
1. Load historical patient volume data from PostgreSQL
2. Preprocess and validate the data
3. Split into train/test sets
4. Train the Prophet model
5. Evaluate and log metrics to MLflow
6. Save the model for serving

================================================================================
TRAINING MODES
================================================================================

This agent supports two training paradigms:

1. LOCAL TRAINING (this module):
   - Uses only this hospital's data
   - Fast iteration and immediate deployment
   - Good for hospital-specific patterns
   - Run via: python train.py or POST /train endpoint

2. FEDERATED TRAINING (fl_client.py):
   - Collaborative learning across hospital network
   - Privacy-preserving (no raw data shared)
   - Better generalization across demographics
   - Run via: python fl_client.py

================================================================================
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from config import settings
from model import DemandForecaster
from mlflow_tracking import MLflowTracker, log_training_run

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles data loading from PostgreSQL for training.
    
    This class encapsulates database connectivity and query logic,
    providing clean interfaces for fetching training data.
    """
    
    def __init__(self, database_url: Optional[str] = None) -> None:
        """
        Initialize the data loader.
        
        Args:
            database_url: PostgreSQL connection string (default from settings)
        """
        self.database_url = str(database_url or settings.database_url)
        self._engine: Optional[Engine] = None
    
    @property
    def engine(self) -> Engine:
        """Lazily create database engine."""
        if self._engine is None:
            self._engine = create_engine(
                self.database_url,
                pool_size=settings.db_pool_size,
                pool_timeout=settings.db_pool_timeout,
                pool_pre_ping=True,  # Verify connections before use
            )
        return self._engine
    
    def load_patient_volume(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        department: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load patient volume data from the database.
        
        Args:
            start_date: Start of date range (default: 2 years ago)
            end_date: End of date range (default: now)
            department: Filter by department (ER, ICU, etc.)
        
        Returns:
            DataFrame with columns: timestamp, patient_count, department
        """
        # Default to 2 years of historical data
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=730))
        
        # Build query
        query = text("""
            SELECT 
                DATE_TRUNC('hour', arrival_time) as timestamp,
                COUNT(*) as patient_count,
                department
            FROM patient_arrivals
            WHERE arrival_time BETWEEN :start_date AND :end_date
            {department_filter}
            GROUP BY DATE_TRUNC('hour', arrival_time), department
            ORDER BY timestamp
        """.format(
            department_filter="AND department = :department" if department else ""
        ))
        
        params = {
            "start_date": start_date,
            "end_date": end_date,
        }
        if department:
            params["department"] = department
        
        try:
            df = pd.read_sql(query, self.engine, params=params)
            logger.info(
                f"Loaded {len(df)} records from database",
                extra={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "department": department,
                }
            )
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def load_aggregated_volume(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load aggregated patient volume across all departments.
        
        This is the primary data source for demand forecasting,
        as it captures total hospital load regardless of department.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
        
        Returns:
            DataFrame with columns: timestamp, patient_count
        """
        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=730))
        
        query = text("""
            SELECT 
                DATE_TRUNC('hour', arrival_time) as timestamp,
                COUNT(*) as patient_count
            FROM patient_arrivals
            WHERE arrival_time BETWEEN :start_date AND :end_date
            GROUP BY DATE_TRUNC('hour', arrival_time)
            ORDER BY timestamp
        """)
        
        try:
            df = pd.read_sql(
                query, 
                self.engine, 
                params={"start_date": start_date, "end_date": end_date}
            )
            
            # Fill missing hours with zeros (no arrivals)
            if not df.empty:
                df = self._fill_missing_hours(df, start_date, end_date)
            
            logger.info(f"Loaded {len(df)} hourly records")
            return df
            
        except SQLAlchemyError as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def _fill_missing_hours(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fill gaps in hourly data with zero counts.
        
        Prophet handles missing data, but explicit zeros are better
        for hours where we know there were no arrivals.
        """
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Create complete hourly index
        full_range = pd.date_range(
            start=start_date.replace(minute=0, second=0, microsecond=0),
            end=end_date.replace(minute=0, second=0, microsecond=0),
            freq="H",
        )
        
        # Reindex with full range, filling missing with 0
        df = df.set_index("timestamp").reindex(full_range, fill_value=0).reset_index()
        df.columns = ["timestamp", "patient_count"]
        
        return df
    
    def generate_synthetic_data(
        self,
        n_days: int = 365,
        base_hourly_rate: float = 25.0,
    ) -> pd.DataFrame:
        """
        Generate synthetic patient volume data for testing/demo.
        
        This method creates realistic-looking hospital demand data
        with proper seasonality patterns when real data is unavailable.
        
        Args:
            n_days: Number of days of data to generate
            base_hourly_rate: Average hourly patient count
        
        Returns:
            DataFrame with synthetic patient volume data
        """
        logger.info(f"Generating {n_days} days of synthetic data")
        
        # Generate hourly timestamps
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=n_days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq="H")
        
        n_hours = len(timestamps)
        
        # Base trend (slight growth over time)
        trend = np.linspace(0, 0.1, n_hours) * base_hourly_rate
        
        # Yearly seasonality (flu season peak in winter)
        day_of_year = np.array([t.dayofyear for t in timestamps])
        yearly = 0.2 * base_hourly_rate * np.sin(2 * np.pi * (day_of_year - 30) / 365)
        
        # Weekly seasonality (weekend peaks for ER)
        day_of_week = np.array([t.dayofweek for t in timestamps])
        weekly = 0.15 * base_hourly_rate * (
            np.where(day_of_week >= 5, 1, -0.5)  # Higher on weekends
        )
        
        # Daily seasonality (evening peak)
        hour_of_day = np.array([t.hour for t in timestamps])
        daily = 0.3 * base_hourly_rate * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Random noise
        noise = np.random.normal(0, 0.1 * base_hourly_rate, n_hours)
        
        # Combine components
        patient_counts = (
            base_hourly_rate + trend + yearly + weekly + daily + noise
        ).clip(min=0).round().astype(int)
        
        # Add some holiday spikes
        for ts_idx, ts in enumerate(timestamps):
            # Christmas spike
            if ts.month == 12 and ts.day in [24, 25, 26]:
                patient_counts[ts_idx] = int(patient_counts[ts_idx] * 1.3)
            # July 4th spike (trauma)
            if ts.month == 7 and ts.day == 4:
                patient_counts[ts_idx] = int(patient_counts[ts_idx] * 1.4)
        
        return pd.DataFrame({
            "timestamp": timestamps,
            "patient_count": patient_counts,
        })
    
    def close(self) -> None:
        """Close database connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None


def split_train_test(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets.
    
    For time series, we use a temporal split (last N days for testing)
    rather than random split to prevent data leakage.
    
    Args:
        df: Full dataset with timestamp column
        test_days: Number of days to reserve for testing
    
    Returns:
        Tuple of (train_df, test_df)
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    split_date = df["timestamp"].max() - timedelta(days=test_days)
    
    train_df = df[df["timestamp"] < split_date]
    test_df = df[df["timestamp"] >= split_date]
    
    logger.info(
        f"Split data: {len(train_df)} training, {len(test_df)} test samples",
        extra={
            "split_date": split_date.isoformat(),
            "test_days": test_days,
        }
    )
    
    return train_df, test_df


def train_model(
    use_synthetic: bool = False,
    test_days: int = 30,
    register_model: bool = True,
    department: Optional[str] = None,
) -> Tuple[DemandForecaster, Dict[str, float], str]:
    """
    Execute the full training pipeline.
    
    This is the main entry point for local training. It:
    1. Loads data (real or synthetic)
    2. Splits into train/test
    3. Trains the Prophet model
    4. Evaluates on test set
    5. Logs everything to MLflow
    
    Args:
        use_synthetic: Use synthetic data instead of database
        test_days: Days to hold out for testing
        register_model: Register in MLflow Model Registry
        department: Optional department filter
    
    Returns:
        Tuple of (trained_model, metrics_dict, mlflow_run_id)
    """
    logger.info("Starting training pipeline")
    
    # Load data
    loader = DataLoader()
    
    try:
        if use_synthetic:
            logger.info("Using synthetic data for training")
            df = loader.generate_synthetic_data(n_days=365)
        else:
            logger.info("Loading data from database")
            if department:
                df = loader.load_patient_volume(department=department)
            else:
                df = loader.load_aggregated_volume()
        
        if df.empty:
            raise ValueError("No data available for training")
        
        # Split data
        train_df, test_df = split_train_test(df, test_days=test_days)
        
        # Initialize and train model
        forecaster = DemandForecaster()
        forecaster.fit(train_df)
        
        # Evaluate on test set
        metrics = forecaster.evaluate(test_df)
        
        logger.info(
            "Training complete",
            extra={
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
            }
        )
        
        # Log to MLflow
        run_name = f"train_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        if department:
            run_name = f"{run_name}_{department}"
        
        run_id = log_training_run(
            forecaster=forecaster,
            metrics=metrics,
            run_name=run_name,
            register_model=register_model,
        )
        
        logger.info(f"Training run logged to MLflow: {run_id}")
        
        return forecaster, metrics, run_id
        
    finally:
        loader.close()


def train_and_save_local(
    output_path: str = "/models/demand_forecast_latest.json",
    use_synthetic: bool = False,
) -> None:
    """
    Train and save model locally (without MLflow).
    
    Useful for quick iterations or when MLflow is unavailable.
    
    Args:
        output_path: Path to save the model
        use_synthetic: Use synthetic data
    """
    loader = DataLoader()
    
    try:
        if use_synthetic:
            df = loader.generate_synthetic_data()
        else:
            df = loader.load_aggregated_volume()
        
        train_df, test_df = split_train_test(df)
        
        forecaster = DemandForecaster()
        forecaster.fit(train_df)
        
        metrics = forecaster.evaluate(test_df)
        logger.info(f"Model metrics: {metrics}")
        
        forecaster.save(output_path)
        logger.info(f"Model saved to {output_path}")
        
    finally:
        loader.close()


if __name__ == "__main__":
    """
    Command-line interface for training.
    
    Usage:
        python train.py [--synthetic] [--no-register] [--department DEPT]
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Train demand forecast model")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data instead of database",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="Don't register model in MLflow registry",
    )
    parser.add_argument(
        "--department",
        type=str,
        default=None,
        help="Filter by department (e.g., ER, ICU)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=30,
        help="Number of days to hold out for testing",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Save locally without MLflow",
    )
    
    args = parser.parse_args()
    
    if args.local_only:
        train_and_save_local(use_synthetic=args.synthetic)
    else:
        model, metrics, run_id = train_model(
            use_synthetic=args.synthetic,
            test_days=args.test_days,
            register_model=not args.no_register,
            department=args.department,
        )
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"MLflow Run ID: {run_id}")
        print(f"MAE: {metrics['mae']:.2f} patients")
        print(f"RMSE: {metrics['rmse']:.2f} patients")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"80% Coverage: {metrics['coverage_80pct']:.1f}%")
        print(f"{'='*60}")
