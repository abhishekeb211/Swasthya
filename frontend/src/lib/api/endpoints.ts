import {
  orchestratorApi,
  forecastApi,
  staffApi,
  erOrApi,
  dischargeApi,
  triageApi,
  flApi1,
  flApi2,
} from './client'

// Types
export interface AgentHealth {
  agent: string
  status: 'healthy' | 'unhealthy' | 'unknown'
  lastCheck: string
  responseTime?: number
}

export interface ForecastData {
  date: string
  predicted: number
  upper_bound: number
  lower_bound: number
}

export interface TriageRequest {
  symptoms: string[]
  vitals: {
    temperature?: number
    blood_pressure?: string
    heart_rate?: number
    oxygen_saturation?: number
  }
  lab_readings?: {
    [key: string]: number
  }
}

export interface TriageResponse {
  acuity_level: number
  explanation: string
  recommended_action: string
}

export interface Patient {
  id: string
  name: string
  age: number
  acuity_level: number
  arrival_time: string
  status: string
}

export interface StaffMember {
  id: string
  name: string
  role: string
  department: string
  availability: boolean
}

export interface Schedule {
  staff_id: string
  shifts: Array<{
    date: string
    start_time: string
    end_time: string
  }>
}

export interface DischargeAnalysis {
  patient_id: string
  readiness_score: number
  estimated_discharge_date: string
  explanation: string
  status: 'ready' | 'not_ready' | 'needs_review'
}

export interface FLRound {
  round_id: string
  status: string
  participants: number
  metrics: {
    accuracy?: number
    loss?: number
  }
  timestamp: string
}

// Orchestrator endpoints
export const orchestratorEndpoints = {
  getAgentHealth: () => orchestratorApi.get<AgentHealth[]>('/api/agents/health'),
  runDailyWorkflow: () => orchestratorApi.post('/api/workflow/daily'),
  runForecast: () => orchestratorApi.post('/api/forecast/run'),
  triage: (data: TriageRequest) => orchestratorApi.post<TriageResponse>('/api/triage', data),
}

// Forecast agent endpoints
export const forecastEndpoints = {
  predict: (days: number = 7) => forecastApi.get<ForecastData[]>(`/predict?days=${days}`),
  train: (formData: FormData) =>
    forecastApi.post('/train', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),
}

// Triage agent endpoints
export const triageEndpoints = {
  triage: (data: TriageRequest) => triageApi.post<TriageResponse>('/triage', data),
  batchTriage: (data: TriageRequest[]) => triageApi.post<TriageResponse[]>('/batch-triage', data),
}

// ER/OR agent endpoints
export const erOrEndpoints = {
  addPatient: (data: Partial<Patient>) => erOrApi.post<Patient>('/er/add-patient', data),
  getNextPatient: () => erOrApi.get<Patient>('/er/next-patient'),
  getERQueue: () => erOrApi.get<Patient[]>('/er/queue'),
  scheduleOR: (data: { surgeries: Array<{ id: string; duration: number; priority: number }> }) =>
    erOrApi.post('/or/schedule', data),
  getORSchedule: () => erOrApi.get('/or/schedule'),
}

// Staff agent endpoints
export const staffEndpoints = {
  getStaff: () => staffApi.get<StaffMember[]>('/staff'),
  generateSchedule: (data: { start_date: string; end_date: string }) =>
    staffApi.post<Schedule[]>('/schedule', data),
  getSchedule: (startDate: string, endDate: string) =>
    staffApi.get<Schedule[]>(`/schedule?start_date=${startDate}&end_date=${endDate}`),
}

// Discharge agent endpoints
export const dischargeEndpoints = {
  analyzeAll: () => dischargeApi.get<DischargeAnalysis[]>('/analyze'),
  analyzeSingle: (patientId: string) =>
    dischargeApi.get<DischargeAnalysis>(`/analyze-single?patient_id=${patientId}`),
}

// Federated Learning endpoints
export const flEndpoints = {
  startRound: (server: 1 | 2) => {
    const api = server === 1 ? flApi1 : flApi2
    return api.post<FLRound>('/fl/start-round')
  },
  getStatus: (server: 1 | 2) => {
    const api = server === 1 ? flApi1 : flApi2
    return api.get<FLRound>('/fl/status')
  },
  getHistory: (server: 1 | 2) => {
    const api = server === 1 ? flApi1 : flApi2
    return api.get<FLRound[]>('/fl/history')
  },
  getClients: (server: 1 | 2) => {
    const api = server === 1 ? flApi1 : flApi2
    return api.get<{ client_id: string; status: string }[]>('/fl/clients')
  },
}

