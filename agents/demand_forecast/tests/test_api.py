"""
Demand Forecast Agent - API Unit Tests

Tests for the FastAPI endpoints using TestClient.
Run with: pytest tests/test_api.py -v
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Import the FastAPI app
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import app, app_state


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_forecaster():
    """Create a mock forecaster for testing predictions."""
    mock = MagicMock()
    mock.is_fitted = True
    mock.get_params.return_value = {"seasonality_mode": "multiplicative"}
    mock.training_metadata = {"trained_at": "2024-01-01T00:00:00"}
    
    # Mock prediction output
    import pandas as pd
    mock_predictions = pd.DataFrame({
        "ds": pd.date_range(start="2024-01-15", periods=24, freq="H"),
        "yhat": [50 + i for i in range(24)],
        "yhat_lower": [40 + i for i in range(24)],
        "yhat_upper": [60 + i for i in range(24)],
    })
    mock.predict.return_value = mock_predictions
    
    return mock


class TestHealthEndpoint:
    """Tests for the /health endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_returns_valid_structure(self, client):
        """Health endpoint should return expected JSON structure."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "timestamp" in data
        assert "checks" in data
    
    def test_health_check_contains_model_status(self, client):
        """Health check should include model availability status."""
        response = client.get("/health")
        data = response.json()
        
        assert "model" in data["checks"]
        assert "status" in data["checks"]["model"]


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""
    
    def test_predict_without_model_returns_503(self, client):
        """Prediction should fail gracefully when no model is loaded."""
        # Ensure no model is loaded
        app_state.forecaster = None
        
        response = client.post(
            "/predict",
            json={"horizon_hours": 24}
        )
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert data["detail"]["error"] == "model_not_available"
    
    def test_predict_with_valid_request(self, client, mock_forecaster):
        """Prediction should succeed with valid request and loaded model."""
        # Inject mock forecaster
        app_state.forecaster = mock_forecaster
        app_state.model_version = "v1.0-test"
        app_state.model_loaded_at = datetime.utcnow()
        
        response = client.post(
            "/predict",
            json={"horizon_hours": 24}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "request_id" in data
        assert "model_version" in data
        assert "predictions" in data
        assert len(data["predictions"]) == 24
        
        # Clean up
        app_state.forecaster = None
    
    def test_predict_validates_horizon_hours(self, client, mock_forecaster):
        """Prediction should validate horizon_hours bounds."""
        app_state.forecaster = mock_forecaster
        
        # Test minimum bound
        response = client.post(
            "/predict",
            json={"horizon_hours": 0}
        )
        assert response.status_code == 422  # Validation error
        
        # Test maximum bound
        response = client.post(
            "/predict",
            json={"horizon_hours": 1000}
        )
        assert response.status_code == 422  # Validation error
        
        # Clean up
        app_state.forecaster = None
    
    def test_predict_with_optional_parameters(self, client, mock_forecaster):
        """Prediction should accept optional parameters."""
        app_state.forecaster = mock_forecaster
        app_state.model_version = "v1.0-test"
        app_state.model_loaded_at = datetime.utcnow()
        
        response = client.post(
            "/predict",
            json={
                "horizon_hours": 48,
                "start_datetime": "2024-01-15T00:00:00Z",
                "include_components": False
            }
        )
        
        assert response.status_code == 200
        
        # Clean up
        app_state.forecaster = None


class TestTrainEndpoint:
    """Tests for the /train endpoint."""
    
    def test_train_returns_job_id(self, client):
        """Training endpoint should return a job ID."""
        response = client.post(
            "/train",
            json={"use_synthetic_data": True}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "job_id" in data
        assert "status" in data
        assert data["status"] == "pending"
    
    def test_train_validates_test_days(self, client):
        """Training should validate test_days bounds."""
        # Test minimum bound
        response = client.post(
            "/train",
            json={"test_days": 3}  # Below minimum of 7
        )
        assert response.status_code == 422
        
        # Test maximum bound
        response = client.post(
            "/train",
            json={"test_days": 100}  # Above maximum of 90
        )
        assert response.status_code == 422


class TestTrainingStatus:
    """Tests for the /train/{job_id} endpoint."""
    
    def test_get_nonexistent_job_returns_404(self, client):
        """Fetching non-existent training job should return 404."""
        response = client.get("/train/nonexistent-job-id")
        assert response.status_code == 404


class TestModelEndpoints:
    """Tests for model management endpoints."""
    
    def test_model_info_without_model(self, client):
        """Model info should indicate no model loaded."""
        app_state.forecaster = None
        
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert data["loaded"] == False
    
    def test_model_refresh_endpoint_exists(self, client):
        """Model refresh endpoint should be accessible."""
        response = client.post("/model/refresh")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestRequestValidation:
    """Tests for request validation."""
    
    def test_invalid_json_returns_422(self, client):
        """Invalid JSON should return validation error."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_content_type_handled(self, client):
        """Requests without proper content type should be handled."""
        response = client.post("/predict")
        # Should either use defaults or return validation error
        assert response.status_code in [200, 422, 503]


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
