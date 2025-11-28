# Swasthya: India's Decentralized Health Intelligence Network

![Static Badge](https://img.shields.io/badge/status-in%20progress-blue) ![Static Badge](https://img.shields.io/badge/tech-Blockchain%2C_AI%2C_IoT-purple) ![Static Badge](https://img.shields.io/badge/services-16-green) ![Static Badge](https://img.shields.io/badge/FL-enabled-orange)

---

## 🚀 Quick Start

Get the entire Swasthya platform running in minutes.

### Prerequisites

- Docker & Docker Compose (v2.0+)
- 8GB RAM minimum (16GB recommended)
- Ports 3000, 5000, 5432, 8001-8005, 9090-9091 available

### 1. Clone & Configure

```bash
# Clone the repository
git clone https://github.com/your-org/swasthya.git
cd swasthya

# Copy environment template and configure
cp .env.example .env
```

### 2. Start All Services

```bash
# Launch the entire platform (16 services)
docker-compose up -d

# Watch logs (optional)
docker-compose logs -f
```

### 3. Verify Health

```bash
# Check orchestrator and all agents
curl http://localhost:3000/api/agents/health
```

Expected response:
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "agents": {
    "demand_forecast": { "status": "healthy", "url": "http://swasthya-demand-forecast:8001" },
    "staff_scheduling": { "status": "healthy", "url": "http://swasthya-staff-scheduling:8002" },
    "eror_scheduling": { "status": "healthy", "url": "http://swasthya-eror-scheduling:8003" },
    "discharge_planning": { "status": "healthy", "url": "http://swasthya-discharge-planning:8004" },
    "triage_acuity": { "status": "healthy", "url": "http://swasthya-triage-acuity:8005" }
  }
}
```

### 4. Trigger a Federated Learning Round

```bash
# Start a federated learning round for demand forecasting
docker-compose up fl-demand-server fl-demand-client-1 fl-demand-client-2 fl-demand-client-3

# Start a federated learning round for triage classification
docker-compose up fl-triage-server fl-triage-client-1 fl-triage-client-2 fl-triage-client-3
```

### 5. Run Demand Forecast

```bash
# Trigger a manual forecast
curl -X POST http://localhost:3000/api/forecast/run

# Or call the agent directly
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"horizon_hours": 168}'
```

### 6. Access MLflow UI

Open [http://localhost:5000](http://localhost:5000) to view experiment tracking, model registry, and metrics.

---

## 🧪 Running Tests

### Orchestrator Tests (Jest)

```bash
cd orchestrator
npm install
npm test
```

### Demand Forecast Tests (Pytest)

```bash
cd agents/demand_forecast
pip install -r requirements.txt
pytest tests/ -v
```

---

## 🔐 Security

Enable API key authentication by setting `API_SECRET_KEY` in your `.env` file:

```bash
API_SECRET_KEY=your-secure-api-key-here
```

Then include the key in all requests:

```bash
curl -H "x-api-key: your-secure-api-key-here" \
  http://localhost:3000/api/agents/health
```

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SWASTHYA PLATFORM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ORCHESTRATOR (Node.js)                    │   │
│  │                     Port: 3000                               │   │
│  │  • Supervisor Workflow    • Agent Coordination               │   │
│  │  • Cron Scheduling        • API Gateway                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│         ┌────────────────────┼────────────────────┐                │
│         ▼                    ▼                    ▼                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │   Demand    │     │    Staff    │     │  ER/OR      │          │
│  │  Forecast   │     │  Scheduling │     │ Scheduling  │          │
│  │   :8001     │     │    :8002    │     │   :8003     │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│         │                                       │                  │
│         ▼                    ▼                    ▼                │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │  Discharge  │     │   Triage    │     │   MLflow    │          │
│  │  Planning   │     │   Acuity    │     │   Server    │          │
│  │    :8004    │     │    :8005    │     │    :5000    │          │
│  └─────────────┘     └─────────────┘     └─────────────┘          │
│                                                 │                  │
│  ┌─────────────────────────────────────────────┴───────────────┐  │
│  │                    FEDERATED LEARNING                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │  │
│  │  │ FL-Demand    │  │ FL-Triage    │  │  Hospital    │       │  │
│  │  │   Server     │  │   Server     │  │  Clients     │       │  │
│  │  │   :9090      │  │   :9091      │  │   (1,2,3)    │       │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              │                                     │
│                              ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                     PostgreSQL :5432                         │  │
│  │              (Patients, Admissions, Schedules)               │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🩺 Project Overview

Swasthya (meaning "Health" in Sanskrit) is a revolutionary decentralized healthcare intelligence network that addresses critical gaps in India's healthcare infrastructure through cutting-edge technology.

### 🎯 Vision

To create a unified, intelligent healthcare ecosystem that empowers patients, optimizes hospital operations, and enables real-time national health intelligence.

### 🔍 Problem Statement

India's healthcare system faces three critical challenges:
* **Fragmented Medical Records:** Patient data is scattered across different hospitals and is often inaccessible when needed most.
* **Delayed Emergency Response:** Inefficient coordination between ambulances, hospitals, and specialists during critical care situations.
* **Manual Hospital Operations:** Suboptimal resource allocation, staff scheduling, and patient flow management lead to inefficiencies.
* **Limited Real-time Insights:** Inadequate public health monitoring prevents proactive responses to disease outbreaks and health trends.

---

## 💡 Solution Architecture

### Core Technological Pillars

| Component | Technology Stack | Purpose |
| :--- | :--- | :--- |
| **Digital Health Wallets** | Blockchain + Aadhaar Integration | Patient-controlled, secure, and interoperable health records. |
| **AI Command Centers** | NVIDIA Clara + Meta LLaMA | Predictive diagnostics, operational forecasting, and workflow automation. |
| **Real-time Monitoring** | IoT Devices | Continuous vital signs tracking for at-risk patients and remote care. |
| **Privacy-Preserving AI**| Federated Learning | Distributed model training on hospital data without centralizing sensitive information. |

---

## 🤖 Intelligent Agent Ecosystem

### 🎯 Demand Forecast Agent
* **Tech Stack:** Meta Kats + PyTorch Forecasting (TFT models)
* **Capabilities:** Time-series analysis for patient admission rates, anomaly detection for potential outbreaks, and seasonal forecasting for resource planning.
* **Advantage:** Better interpretability and performance on complex time-series data compared to traditional LSTMs.

### 👥 Staff Scheduling Agent (RL)
* **Tech Stack:** Reinforcement Learning + Meta Code Llama
* **Innovation:** The agent not only creates optimal schedules but also uses Code Llama to generate human-readable explanations for its decisions.
* **Benefit:** Enhanced transparency and trust among hospital staff.

### 🚑 Triage & Acuity Agent
* **Foundation:** NVIDIA CLARA framework
* **Features:** Utilizes pre-trained medical imaging models (e.g., for X-rays, CT scans) for rapid initial assessment.
* **Deployment:** Served via NVIDIA Triton Inference Server for real-time, low-latency inference at the point of care.

### 🏥 ER/OR Scheduling Agent
* **Hybrid Approach:**
    * **NVIDIA RAPIDS XGBoost:** Predicts surgery duration based on historical data.
    * **GPU-accelerated RL:** Dynamically reschedules operating rooms in real-time as emergencies arise.
* **Performance:** Massively parallel processing on GPUs enables real-time optimization of complex schedules.

### 📋 Discharge Planning Agent
* **Components:**
    * **NVIDIA CLARA Model Zoo:** Clinical prediction models to identify patients ready for discharge.
    * **Meta Llama 2/3:** Acts as a discharge assistant, analyzing charts, tracking milestones, and generating discharge summaries.
* **Function:** Automates routine discharge tasks to free up clinical staff.

### 🎮 Supervisor Agent (Central Orchestrator)
* **Brain:** Fine-tuned Meta Llama 2/3 Reasoning Engine
* **Role:** Coordinates the multi-agent system, processes complex queries, and manages negotiations between agents (e.g., balancing ER demand with OR availability).

---

## 🛠 Technical Implementation

### Blockchain Layer

A simplified smart contract for managing patient data access.

```solidity
// Health Wallet Smart Contract
contract HealthWallet {
    struct HealthRecord {
        string recordHash;
        uint256 timestamp;
        address provider;
    }

    mapping(address => HealthRecord[]) private patientRecords;
    mapping(address => mapping(address => bool)) private authorizedEntities;

    event AccessGranted(address indexed patient, address indexed provider);
    event AccessRevoked(address indexed patient, address indexed provider);

    function grantAccess(address provider) public {
        authorizedEntities[msg.sender][provider] = true;
        emit AccessGranted(msg.sender, provider);
    }

    function revokeAccess(address provider) public {
        authorizedEntities[msg.sender][provider] = false;
        emit AccessRevoked(msg.sender, provider);
    }
}
```

---

## 📁 Project Structure

```
swasthya/
├── agents/                    # AI Microservices
│   ├── demand_forecast/       # Patient volume prediction
│   ├── discharge_planning/    # Discharge automation
│   └── eror_scheduling/       # ER/OR optimization
├── orchestrator/              # Node.js Supervisor
│   ├── src/
│   │   ├── middleware/        # Auth & security
│   │   ├── routes/            # API endpoints
│   │   ├── scheduler/         # Cron jobs
│   │   └── supervisor/        # Workflow coordination
│   └── tests/                 # Jest tests
├── federated_learning/        # FL Server & Clients
├── datasets/                  # Simulation data for FL
├── data/                      # SQL initialization
├── docker-compose.yml         # Full stack deployment
└── README.md                  # This file
```

---

## 🌐 API Reference

### Orchestrator Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/agents/health` | Health check for all agents |
| POST | `/api/workflow/daily` | Trigger daily optimization workflow |
| POST | `/api/forecast/run` | Run demand forecast manually |

### Demand Forecast Agent

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |
| POST | `/predict` | Generate patient volume forecast |
| POST | `/train` | Trigger model retraining |
| GET | `/train/{job_id}` | Check training job status |
| GET | `/model/info` | Get loaded model information |
| POST | `/model/refresh` | Reload model from registry |

---

## 📝 Environment Variables

Create a `.env` file from `.env.example`:

```bash
# Database
POSTGRES_USER=swasthya
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=swasthya_db
POSTGRES_HOST=swasthya-postgres
POSTGRES_PORT=5432

# MLflow
MLFLOW_TRACKING_URI=http://swasthya-mlflow:5000
MLFLOW_DB_USER=mlflow
MLFLOW_DB_PASSWORD=mlflow-password
MLFLOW_DB_NAME=mlflow_db
MLFLOW_PORT=5000

# Orchestrator
ORCHESTRATOR_PORT=3000
API_SECRET_KEY=your-api-secret-key

# Agent Ports
PORT_DEMAND_FORECAST=8001
PORT_STAFF_SCHEDULING=8002
PORT_EROR_SCHEDULING=8003
PORT_DISCHARGE_PLANNING=8004
PORT_TRIAGE_ACUITY=8005

# Federated Learning
FL_DEMAND_PORT=9090
FL_TRIAGE_PORT=9091
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

---

<p align="center">
  <strong>Built with ❤️ for India's Healthcare Future</strong>
</p>
