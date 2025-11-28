"""
Federated Learning Client - Serialization/Deserialization Utilities

This module provides utilities for converting ML model objects to byte arrays
and back for transmission over the network in federated learning.

================================================================================
WHY SERIALIZATION IS CRITICAL FOR FL
================================================================================

In federated learning, model parameters must be transmitted between:
- Clients → Server: After local training
- Server → Clients: Before local training (global model distribution)

Different ML frameworks store parameters differently:

    Prophet (demand forecasting):
    ├── JSON serialization (native)
    ├── Parameters: trend (k, m, delta), seasonality (beta)
    └── Challenge: Complex nested structure

    XGBoost (triage prediction):
    ├── Binary format (.ubj) or JSON
    ├── Parameters: Tree structures + leaf weights
    └── Challenge: Large model size
    
    scikit-learn:
    ├── Pickle serialization
    ├── Parameters: Model-specific
    └── Challenge: Version compatibility

This module provides a unified interface for all model types,
handling the complexity of serialization behind simple APIs.

================================================================================
SERIALIZATION FORMAT
================================================================================

We use a two-layer approach:

Layer 1: Model-specific serialization
└── Convert model objects to their native format (JSON, binary, pickle)

Layer 2: FL transport format
└── Convert to numpy arrays (Flower's native transport format)

    ┌─────────────────────┐
    │    Prophet Model    │
    │   (Python object)   │
    └──────────┬──────────┘
               │
               ▼ serialize_prophet()
    ┌─────────────────────┐
    │    JSON String      │
    └──────────┬──────────┘
               │
               ▼ string_to_ndarray()
    ┌─────────────────────┐
    │   np.ndarray        │
    │   (uint8 bytes)     │
    └──────────┬──────────┘
               │
               ▼ Flower gRPC transport
    ┌─────────────────────┐
    │    FL Server        │
    └─────────────────────┘

================================================================================
"""

from __future__ import annotations

import json
import logging
import pickle
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# CORE SERIALIZATION UTILITIES
# =============================================================================

def string_to_ndarray(s: str) -> np.ndarray:
    """
    Convert a string to a numpy array of bytes.
    
    This is the standard format for transmitting non-numeric data
    (like JSON) through Flower's parameter system.
    
    Args:
        s: String to convert
    
    Returns:
        numpy array of uint8 (bytes)
    
    Example:
        >>> arr = string_to_ndarray('{"key": "value"}')
        >>> arr.dtype
        dtype('uint8')
    """
    return np.frombuffer(s.encode('utf-8'), dtype=np.uint8)


def ndarray_to_string(arr: np.ndarray) -> str:
    """
    Convert a numpy array of bytes back to a string.
    
    Args:
        arr: numpy array of uint8
    
    Returns:
        Decoded string
    
    Example:
        >>> arr = string_to_ndarray('hello')
        >>> ndarray_to_string(arr)
        'hello'
    """
    return arr.tobytes().decode('utf-8')


def bytes_to_ndarray(b: bytes) -> np.ndarray:
    """
    Convert raw bytes to a numpy array.
    
    Args:
        b: Raw bytes
    
    Returns:
        numpy array of uint8
    """
    return np.frombuffer(b, dtype=np.uint8)


def ndarray_to_bytes(arr: np.ndarray) -> bytes:
    """
    Convert a numpy array to raw bytes.
    
    Args:
        arr: numpy array
    
    Returns:
        Raw bytes
    """
    return arr.tobytes()


# =============================================================================
# COMPRESSION UTILITIES
# =============================================================================

def compress_ndarray(arr: np.ndarray, level: int = 6) -> np.ndarray:
    """
    Compress a numpy array using zlib.
    
    Useful for reducing bandwidth when transmitting large models.
    Compression ratio depends on model type:
    - Prophet JSON: ~60-80% size reduction
    - XGBoost binary: ~30-50% size reduction
    
    Args:
        arr: Array to compress
        level: Compression level (1-9, higher = better compression, slower)
    
    Returns:
        Compressed array with header
    
    Example:
        >>> original = string_to_ndarray('x' * 10000)
        >>> compressed = compress_ndarray(original)
        >>> len(compressed) < len(original)
        True
    """
    # Compress
    compressed_bytes = zlib.compress(arr.tobytes(), level)
    
    # Create header with original shape info
    header = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "compressed": True,
    }
    header_json = json.dumps(header).encode('utf-8')
    header_len = len(header_json)
    
    # Pack: [4-byte header length][header JSON][compressed data]
    packed = (
        header_len.to_bytes(4, 'little') +
        header_json +
        compressed_bytes
    )
    
    return np.frombuffer(packed, dtype=np.uint8)


def decompress_ndarray(arr: np.ndarray) -> np.ndarray:
    """
    Decompress an array compressed with compress_ndarray().
    
    Args:
        arr: Compressed array
    
    Returns:
        Decompressed array with original shape/dtype
    """
    data = arr.tobytes()
    
    # Extract header length
    header_len = int.from_bytes(data[:4], 'little')
    
    # Extract header
    header_json = data[4:4+header_len]
    header = json.loads(header_json.decode('utf-8'))
    
    # Extract and decompress data
    compressed_data = data[4+header_len:]
    decompressed_bytes = zlib.decompress(compressed_data)
    
    # Reconstruct array
    dtype = np.dtype(header["dtype"])
    shape = tuple(header["shape"])
    
    result = np.frombuffer(decompressed_bytes, dtype=dtype)
    
    if shape:
        result = result.reshape(shape)
    
    return result


# =============================================================================
# PROPHET MODEL SERIALIZATION
# =============================================================================

class ProphetSerializer:
    """
    Serializer for Facebook Prophet models.
    
    Prophet models have complex internal state including:
    - Stan model parameters
    - Seasonality components
    - Holiday effects
    - Changepoint locations
    
    We use Prophet's native JSON serialization which captures
    all necessary state for exact reconstruction.
    """
    
    @staticmethod
    def serialize(model: Any) -> np.ndarray:
        """
        Serialize a Prophet model to numpy array.
        
        Args:
            model: Fitted Prophet model instance
        
        Returns:
            numpy array containing serialized model
        
        Example:
            >>> from prophet import Prophet
            >>> m = Prophet().fit(df)
            >>> arr = ProphetSerializer.serialize(m)
        """
        try:
            from prophet.serialize import model_to_json
            
            # Serialize to JSON
            json_str = model_to_json(model)
            
            # Convert to array
            arr = string_to_ndarray(json_str)
            
            # Optionally compress (Prophet models can be large)
            if len(arr) > 100_000:  # >100KB
                arr = compress_ndarray(arr)
                logger.debug(f"Compressed Prophet model: {len(json_str)} -> {len(arr)}")
            
            return arr
            
        except ImportError:
            logger.warning("Prophet not installed, using pickle fallback")
            return ProphetSerializer._pickle_serialize(model)
    
    @staticmethod
    def deserialize(arr: np.ndarray) -> Any:
        """
        Deserialize a Prophet model from numpy array.
        
        Args:
            arr: numpy array from serialize()
        
        Returns:
            Prophet model instance
        """
        try:
            from prophet.serialize import model_from_json
            
            # Check if compressed
            try:
                arr = decompress_ndarray(arr)
            except Exception:
                pass  # Not compressed
            
            # Convert to string
            json_str = ndarray_to_string(arr)
            
            # Deserialize
            return model_from_json(json_str)
            
        except ImportError:
            logger.warning("Prophet not installed, using pickle fallback")
            return ProphetSerializer._pickle_deserialize(arr)
    
    @staticmethod
    def _pickle_serialize(model: Any) -> np.ndarray:
        """Fallback pickle serialization."""
        pickled = pickle.dumps(model)
        return bytes_to_ndarray(pickled)
    
    @staticmethod
    def _pickle_deserialize(arr: np.ndarray) -> Any:
        """Fallback pickle deserialization."""
        return pickle.loads(ndarray_to_bytes(arr))
    
    @staticmethod
    def extract_parameters(model: Any) -> List[np.ndarray]:
        """
        Extract trainable parameters from Prophet model as numpy arrays.
        
        This is used for federated averaging of individual parameters
        rather than full model serialization.
        
        Prophet's trainable parameters include:
        - k: Growth rate
        - m: Offset
        - delta: Changepoint adjustments
        - beta: Seasonality Fourier coefficients
        
        Args:
            model: Fitted Prophet model
        
        Returns:
            List of numpy arrays, one per parameter group
        """
        params = []
        
        if hasattr(model, 'params') and model.params:
            # Growth rate
            if 'k' in model.params:
                params.append(np.array([model.params['k']], dtype=np.float64))
            
            # Offset
            if 'm' in model.params:
                params.append(np.array([model.params['m']], dtype=np.float64))
            
            # Changepoint adjustments
            if 'delta' in model.params:
                delta = model.params['delta']
                if isinstance(delta, list):
                    params.append(np.array(delta, dtype=np.float64))
                else:
                    params.append(np.array([delta], dtype=np.float64))
            
            # Seasonality coefficients
            if 'beta' in model.params:
                beta = model.params['beta']
                if isinstance(beta, list):
                    params.append(np.array(beta, dtype=np.float64))
                else:
                    params.append(np.array([beta], dtype=np.float64))
        
        # Fallback: serialize entire model as single array
        if not params:
            logger.debug("Using full model serialization for Prophet parameters")
            params = [ProphetSerializer.serialize(model)]
        
        return params
    
    @staticmethod
    def apply_parameters(model: Any, params: List[np.ndarray]) -> Any:
        """
        Apply federated parameters to a Prophet model.
        
        Args:
            model: Prophet model instance
            params: List of numpy arrays (from extract_parameters)
        
        Returns:
            Updated Prophet model
        """
        # Check if this is a full serialized model
        if len(params) == 1 and params[0].dtype == np.uint8:
            return ProphetSerializer.deserialize(params[0])
        
        # Apply individual parameters
        if hasattr(model, 'params') and model.params:
            param_idx = 0
            
            if 'k' in model.params and param_idx < len(params):
                model.params['k'] = float(params[param_idx][0])
                param_idx += 1
            
            if 'm' in model.params and param_idx < len(params):
                model.params['m'] = float(params[param_idx][0])
                param_idx += 1
            
            if 'delta' in model.params and param_idx < len(params):
                model.params['delta'] = params[param_idx].tolist()
                param_idx += 1
            
            if 'beta' in model.params and param_idx < len(params):
                model.params['beta'] = params[param_idx].tolist()
                param_idx += 1
        
        return model


# =============================================================================
# XGBOOST MODEL SERIALIZATION
# =============================================================================

class XGBoostSerializer:
    """
    Serializer for XGBoost models.
    
    XGBoost stores model parameters as:
    - Tree structures (splits, thresholds)
    - Leaf values (predictions at each leaf)
    - Model configuration (objective, parameters)
    
    We support multiple serialization formats:
    - JSON: Human-readable, larger size
    - UBJ (Universal Binary JSON): Compact, faster
    - Pickle: Maximum compatibility
    """
    
    @staticmethod
    def serialize(
        model: Any,
        format: str = "json",
    ) -> np.ndarray:
        """
        Serialize an XGBoost model to numpy array.
        
        Args:
            model: XGBoost Booster or sklearn-style model
            format: "json", "ubj", or "pickle"
        
        Returns:
            numpy array containing serialized model
        """
        try:
            import xgboost as xgb
            
            # Handle sklearn-style wrapper
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
            elif isinstance(model, xgb.Booster):
                booster = model
            else:
                # Fallback to pickle
                return XGBoostSerializer._pickle_serialize(model)
            
            if format == "json":
                # JSON serialization
                config = booster.save_config()
                raw_data = booster.save_raw(raw_format='json')
                
                # Combine config and model
                combined = {
                    "config": config,
                    "model": raw_data.decode('utf-8') if isinstance(raw_data, bytes) else raw_data,
                    "format": "json",
                }
                json_str = json.dumps(combined)
                return string_to_ndarray(json_str)
                
            elif format == "ubj":
                # Universal Binary JSON (compact)
                raw_data = booster.save_raw(raw_format='ubj')
                
                # Add format marker
                marker = b'XGB_UBJ:'
                combined = marker + raw_data
                return bytes_to_ndarray(combined)
                
            else:
                return XGBoostSerializer._pickle_serialize(model)
                
        except ImportError:
            logger.warning("XGBoost not installed")
            return XGBoostSerializer._pickle_serialize(model)
    
    @staticmethod
    def deserialize(arr: np.ndarray) -> Any:
        """
        Deserialize an XGBoost model from numpy array.
        
        Args:
            arr: numpy array from serialize()
        
        Returns:
            XGBoost model instance
        """
        try:
            import xgboost as xgb
            
            data = arr.tobytes()
            
            # Check format marker
            if data.startswith(b'XGB_UBJ:'):
                # UBJ format
                raw_data = data[8:]  # Skip marker
                booster = xgb.Booster()
                booster.load_model(bytearray(raw_data))
                return booster
            
            # Try JSON format
            try:
                json_str = ndarray_to_string(arr)
                combined = json.loads(json_str)
                
                if "format" in combined and combined["format"] == "json":
                    booster = xgb.Booster()
                    
                    # Load config first
                    if "config" in combined:
                        booster.load_config(combined["config"])
                    
                    # Load model
                    model_data = combined["model"]
                    if isinstance(model_data, str):
                        model_data = model_data.encode('utf-8')
                    booster.load_model(bytearray(model_data))
                    
                    return booster
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
            
            # Fallback to pickle
            return XGBoostSerializer._pickle_deserialize(arr)
            
        except ImportError:
            return XGBoostSerializer._pickle_deserialize(arr)
    
    @staticmethod
    def _pickle_serialize(model: Any) -> np.ndarray:
        """Pickle serialization fallback."""
        pickled = pickle.dumps(model)
        return bytes_to_ndarray(pickled)
    
    @staticmethod
    def _pickle_deserialize(arr: np.ndarray) -> Any:
        """Pickle deserialization fallback."""
        return pickle.loads(ndarray_to_bytes(arr))
    
    @staticmethod
    def extract_parameters(model: Any) -> List[np.ndarray]:
        """
        Extract XGBoost model parameters as numpy arrays.
        
        For XGBoost, we extract:
        - Tree dump as JSON (structure + splits)
        - Leaf values
        - Feature importances
        
        Args:
            model: XGBoost model
        
        Returns:
            List of numpy arrays
        """
        try:
            import xgboost as xgb
            
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
            else:
                booster = model
            
            # Get model as JSON dump
            model_dump = booster.save_raw(raw_format='json')
            
            # Return as single array (XGBoost trees are hard to decompose)
            return [bytes_to_ndarray(model_dump)]
            
        except Exception as e:
            logger.warning(f"Failed to extract XGBoost parameters: {e}")
            # Fallback to full serialization
            return [XGBoostSerializer.serialize(model)]
    
    @staticmethod
    def apply_parameters(model_template: Any, params: List[np.ndarray]) -> Any:
        """
        Create XGBoost model from federated parameters.
        
        Args:
            model_template: Template model (for config)
            params: Parameters from extract_parameters
        
        Returns:
            New XGBoost model with applied parameters
        """
        try:
            import xgboost as xgb
            
            if not params:
                return model_template
            
            # Reconstruct from raw format
            raw_data = ndarray_to_bytes(params[0])
            
            booster = xgb.Booster()
            booster.load_model(bytearray(raw_data))
            
            return booster
            
        except Exception as e:
            logger.warning(f"Failed to apply XGBoost parameters: {e}")
            return model_template


# =============================================================================
# GENERIC SERIALIZATION INTERFACE
# =============================================================================

class ModelSerializer:
    """
    Unified interface for model serialization across different ML frameworks.
    
    Automatically detects model type and uses appropriate serializer.
    
    Example:
        >>> from prophet import Prophet
        >>> m = Prophet().fit(df)
        >>> arr = ModelSerializer.serialize(m)
        >>> m2 = ModelSerializer.deserialize(arr, "prophet")
    """
    
    SERIALIZERS = {
        "prophet": ProphetSerializer,
        "xgboost": XGBoostSerializer,
    }
    
    @classmethod
    def serialize(cls, model: Any, model_type: Optional[str] = None) -> np.ndarray:
        """
        Serialize any supported model to numpy array.
        
        Args:
            model: Model instance to serialize
            model_type: Optional type hint ("prophet", "xgboost")
        
        Returns:
            numpy array containing serialized model
        """
        # Auto-detect model type
        if model_type is None:
            model_type = cls._detect_model_type(model)
        
        serializer = cls.SERIALIZERS.get(model_type)
        
        if serializer:
            return serializer.serialize(model)
        else:
            # Generic pickle fallback
            logger.warning(f"Unknown model type '{model_type}', using pickle")
            return bytes_to_ndarray(pickle.dumps(model))
    
    @classmethod
    def deserialize(cls, arr: np.ndarray, model_type: str) -> Any:
        """
        Deserialize a model from numpy array.
        
        Args:
            arr: numpy array from serialize()
            model_type: Type of model ("prophet", "xgboost")
        
        Returns:
            Deserialized model instance
        """
        serializer = cls.SERIALIZERS.get(model_type)
        
        if serializer:
            return serializer.deserialize(arr)
        else:
            # Generic pickle fallback
            return pickle.loads(ndarray_to_bytes(arr))
    
    @classmethod
    def extract_parameters(cls, model: Any, model_type: Optional[str] = None) -> List[np.ndarray]:
        """Extract parameters for federated averaging."""
        if model_type is None:
            model_type = cls._detect_model_type(model)
        
        serializer = cls.SERIALIZERS.get(model_type)
        
        if serializer and hasattr(serializer, 'extract_parameters'):
            return serializer.extract_parameters(model)
        else:
            # Fallback: serialize entire model
            return [cls.serialize(model, model_type)]
    
    @classmethod
    def apply_parameters(
        cls,
        model: Any,
        params: List[np.ndarray],
        model_type: Optional[str] = None,
    ) -> Any:
        """Apply federated parameters to a model."""
        if model_type is None:
            model_type = cls._detect_model_type(model)
        
        serializer = cls.SERIALIZERS.get(model_type)
        
        if serializer and hasattr(serializer, 'apply_parameters'):
            return serializer.apply_parameters(model, params)
        else:
            # Fallback: deserialize full model
            if params:
                return cls.deserialize(params[0], model_type)
            return model
    
    @classmethod
    def _detect_model_type(cls, model: Any) -> str:
        """Auto-detect model type from instance."""
        model_class = type(model).__name__
        module = type(model).__module__
        
        if "prophet" in module.lower() or model_class == "Prophet":
            return "prophet"
        elif "xgboost" in module.lower() or "xgb" in module.lower():
            return "xgboost"
        else:
            return "unknown"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def serialize_model(model: Any, model_type: Optional[str] = None) -> np.ndarray:
    """Convenience function for model serialization."""
    return ModelSerializer.serialize(model, model_type)


def deserialize_model(arr: np.ndarray, model_type: str) -> Any:
    """Convenience function for model deserialization."""
    return ModelSerializer.deserialize(arr, model_type)


def get_model_parameters(model: Any, model_type: Optional[str] = None) -> List[np.ndarray]:
    """Extract model parameters for FL."""
    return ModelSerializer.extract_parameters(model, model_type)


def set_model_parameters(
    model: Any,
    params: List[np.ndarray],
    model_type: Optional[str] = None,
) -> Any:
    """Apply FL parameters to model."""
    return ModelSerializer.apply_parameters(model, params, model_type)
