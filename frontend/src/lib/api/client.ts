import axios from 'axios'

const ORCHESTRATOR_URL = import.meta.env.VITE_ORCHESTRATOR_URL || 'http://localhost:3000'
const FORECAST_AGENT_URL = import.meta.env.VITE_FORECAST_AGENT_URL || 'http://localhost:8001'
const STAFF_AGENT_URL = import.meta.env.VITE_STAFF_AGENT_URL || 'http://localhost:8002'
const ER_OR_AGENT_URL = import.meta.env.VITE_ER_OR_AGENT_URL || 'http://localhost:8003'
const DISCHARGE_AGENT_URL = import.meta.env.VITE_DISCHARGE_AGENT_URL || 'http://localhost:8004'
const TRIAGE_AGENT_URL = import.meta.env.VITE_TRIAGE_AGENT_URL || 'http://localhost:8005'
const FL_SERVER_1_URL = import.meta.env.VITE_FL_SERVER_1_URL || 'http://localhost:8086'
const FL_SERVER_2_URL = import.meta.env.VITE_FL_SERVER_2_URL || 'http://localhost:8087'
const MLFLOW_URL = import.meta.env.VITE_MLFLOW_URL || 'http://localhost:5000'

// Orchestrator API client
export const orchestratorApi = axios.create({
  baseURL: ORCHESTRATOR_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Agent API clients
export const forecastApi = axios.create({
  baseURL: FORECAST_AGENT_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const staffApi = axios.create({
  baseURL: STAFF_AGENT_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const erOrApi = axios.create({
  baseURL: ER_OR_AGENT_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const dischargeApi = axios.create({
  baseURL: DISCHARGE_AGENT_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const triageApi = axios.create({
  baseURL: TRIAGE_AGENT_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const flApi1 = axios.create({
  baseURL: FL_SERVER_1_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const flApi2 = axios.create({
  baseURL: FL_SERVER_2_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptors for error handling
const addErrorHandling = (api: typeof orchestratorApi) => {
  api.interceptors.response.use(
    (response) => response,
    (error) => {
      console.error('API Error:', error)
      return Promise.reject(error)
    }
  )
}

;[orchestratorApi, forecastApi, staffApi, erOrApi, dischargeApi, triageApi, flApi1, flApi2].forEach(
  addErrorHandling
)

export { MLFLOW_URL }

