# Federated Learning System for Hospital AI

A privacy-preserving federated learning system that enables multiple hospitals to collaboratively train AI models **without sharing patient data**.

## 🎯 Key Features

- **Data Sovereignty**: Patient data never leaves hospital networks
- **HIPAA/GDPR Compliant**: Only model parameters are transmitted
- **Smart Aggregation**: BestModelStrategy improves on standard FedAvg
- **Robust Clients**: Automatic retry with exponential backoff
- **Dual Model Support**: Demand forecasting (Prophet) and Triage prediction (XGBoost)

## 📊 Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                          FL SERVER                                      │
│                                                                        │
│   • Coordinates training rounds                                         │
│   • Aggregates model PARAMETERS (not data)                             │
│   • Uses BestModelStrategy for intelligent aggregation                 │
│   • Never sees any patient information                                 │
└────────────────────────────────────────────────────────────────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │   Hospital A     │  │   Hospital B     │  │   Hospital C     │
  │                  │  │                  │  │                  │
  │ [Patient Data]   │  │ [Patient Data]   │  │ [Patient Data]   │
  │       ↓          │  │       ↓          │  │       ↓          │
  │ [Local Training] │  │ [Local Training] │  │ [Local Training] │
  │       ↓          │  │       ↓          │  │       ↓          │
  │ Send Parameters→ │  │ Send Parameters→ │  │ Send Parameters→ │
  └──────────────────┘  └──────────────────┘  └──────────────────┘
```

## 🚀 Quick Start

### Using Docker Compose

```bash
# Start the FL server
docker-compose up fl-demand-server

# In separate terminals, start clients
docker-compose up fl-client-hospital-a
docker-compose up fl-client-hospital-b
```

### Manual Start

```bash
# Terminal 1: Start server (waits for clients)
python -m federated_learning.server.demand_server \
    --address 0.0.0.0:8087 \
    --rounds 5 \
    --min-clients 2

# Terminal 2: Start client (Hospital A)
python -m federated_learning.client.demand_client \
    --server localhost:8087 \
    --hospital-id hospital_alpha \
    --data /path/to/patient_volumes.csv

# Terminal 3: Start client (Hospital B)
python -m federated_learning.client.demand_client \
    --server localhost:8087 \
    --hospital-id hospital_beta \
    --data /path/to/patient_volumes.csv
```

## 📁 Project Structure

```
federated_learning/
├── server/
│   ├── __init__.py
│   ├── strategy.py          # BestModelStrategy (custom FedAvg)
│   ├── demand_server.py     # FL server for demand forecasting
│   └── triage_server.py     # FL server for triage prediction
│
├── client/
│   ├── __init__.py
│   ├── serde.py            # Serialization utilities
│   ├── demand_client.py    # Prophet-based demand client
│   └── triage_client.py    # XGBoost-based triage client
│
├── Dockerfile.server       # Server container
├── Dockerfile.client       # Client container
├── requirements-server.txt # Server dependencies
├── requirements-client.txt # Client dependencies
└── README.md
```

## 🔧 Configuration

### Server Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `FL_SERVER_ADDRESS` | `0.0.0.0:8087` | Server bind address |
| `FL_NUM_ROUNDS` | `5` | Number of training rounds |
| `FL_MIN_FIT_CLIENTS` | `2` | Minimum clients for training |
| `FL_MIN_EVAL_CLIENTS` | `2` | Minimum clients for evaluation |
| `FL_MODEL_SAVE_PATH` | `/tmp/fl_model` | Path to save global model |

### Client Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `FL_SERVER_ADDRESS` | `localhost:8087` | Server address |
| `HOSPITAL_ID` | `hospital_<pid>` | Unique hospital identifier |
| `FL_DATA_PATH` | `/data/patient_volumes.csv` | Path to local data |
| `FL_LOCAL_EPOCHS` | `1` | Local training epochs per round |
| `FL_MAX_RETRIES` | `10` | Max connection retries |
| `FL_RETRY_DELAY` | `5.0` | Initial retry delay (seconds) |

## 📈 Model Types

### Demand Forecasting (Port 8087)

- **Model**: Facebook Prophet (time-series)
- **Input**: Historical patient volumes
- **Output**: Hourly patient count predictions
- **Metric**: MAE (Mean Absolute Error)

```python
# Data format (CSV)
timestamp,patient_count
2024-01-01 00:00:00,15
2024-01-01 01:00:00,12
...
```

### Triage Prediction (Port 8086)

- **Model**: XGBoost (classification)
- **Input**: Patient vitals and demographics
- **Output**: Acuity level (ESI 1-5)
- **Metric**: Accuracy

```python
# Data format (CSV)
age,heart_rate,blood_pressure_systolic,...,acuity_level
45,88,120,...,3
67,102,145,...,2
...
```

## 🧠 BestModelStrategy

Our custom aggregation strategy improves on standard Federated Averaging:

### Standard FedAvg
```
weight_k = num_samples_k / total_samples
```

### BestModelStrategy
```
weight_k = α × performance_weight_k + (1-α) × sample_weight_k
         × reliability_multiplier_k
```

**Features**:
- **Performance-weighted**: Better-performing models contribute more
- **Reliability tracking**: Tracks client consistency over rounds
- **Anomaly detection**: Filters corrupted or malicious updates
- **Configurable**: Adjustable performance weight (α)

```python
from federated_learning.server.strategy import BestModelStrategy

strategy = BestModelStrategy(
    min_fit_clients=2,
    performance_weight=0.3,  # 30% performance, 70% sample size
    use_reliability_tracking=True,
    anomaly_threshold=3.0,   # Z-score threshold
)
```

## 🔒 Security Considerations

1. **Data Never Leaves**: Patient data stays within hospital networks
2. **Parameter Inspection**: Only model weights are transmitted
3. **TLS Encryption**: Use TLS for production deployments
4. **Network Isolation**: Consider VPN or private networks
5. **Audit Logging**: All FL operations are logged

## 🐳 Docker Deployment

### Build Images

```bash
# Build server
docker build -f Dockerfile.server -t fl-server .

# Build client
docker build -f Dockerfile.client -t fl-client .
```

### Run with Docker Compose

```yaml
# docker-compose.yml snippet
services:
  fl-demand-server:
    image: fl-server
    ports:
      - "8087:8087"
    environment:
      - FL_NUM_ROUNDS=5
      - FL_MIN_FIT_CLIENTS=2

  fl-client-hospital-a:
    image: fl-client
    volumes:
      - ./data/hospital_a:/data:ro
    environment:
      - HOSPITAL_ID=hospital_alpha
      - FL_SERVER_ADDRESS=fl-demand-server:8087
```

## 📊 Monitoring

The server logs detailed information:

```
2024-01-15 10:00:00 - INFO - Starting FL Demand Server on 0.0.0.0:8087
2024-01-15 10:00:30 - INFO - Round 1: Aggregating 2 client updates
2024-01-15 10:00:30 - INFO - Weights: {'hospital_a': 0.45, 'hospital_b': 0.55}
2024-01-15 10:00:31 - INFO - Training round complete - Aggregated MAE: 5.23
```

## 🧪 Testing

```bash
# Test with synthetic data (no real data needed)
python -m federated_learning.client.demand_client --no-wait

# Run server tests
pytest tests/test_strategy.py

# Run integration test
./scripts/test_fl_integration.sh
```

## 📚 References

- [Flower Documentation](https://flower.dev/docs/)
- [Federated Learning: Strategies for Improving Communication Efficiency](https://arxiv.org/abs/1610.05492)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

## 📝 License

This project is part of the Hospital AI Platform. See LICENSE for details.
