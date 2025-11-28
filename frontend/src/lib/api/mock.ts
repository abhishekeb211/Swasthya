// Mock API responses for local development
// Use this when backend services are not available

export const mockAgentHealth = [
  {
    agent: 'Demand Forecast',
    status: 'healthy' as const,
    lastCheck: new Date().toISOString(),
    responseTime: 45,
  },
  {
    agent: 'Staff Scheduling',
    status: 'healthy' as const,
    lastCheck: new Date().toISOString(),
    responseTime: 32,
  },
  {
    agent: 'ER/OR Scheduling',
    status: 'healthy' as const,
    lastCheck: new Date().toISOString(),
    responseTime: 28,
  },
  {
    agent: 'Discharge Planning',
    status: 'healthy' as const,
    lastCheck: new Date().toISOString(),
    responseTime: 51,
  },
  {
    agent: 'Triage & Acuity',
    status: 'healthy' as const,
    lastCheck: new Date().toISOString(),
    responseTime: 38,
  },
]

export const mockForecast = Array.from({ length: 7 }, (_, i) => {
  const date = new Date()
  date.setDate(date.getDate() + i)
  const base = 50 + Math.random() * 30
  return {
    date: date.toISOString().split('T')[0],
    predicted: Math.round(base),
    upper_bound: Math.round(base * 1.2),
    lower_bound: Math.round(base * 0.8),
  }
})

export const mockERQueue = [
  {
    id: 'p1',
    name: 'John Doe',
    age: 45,
    acuity_level: 5,
    arrival_time: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
    status: 'waiting',
  },
  {
    id: 'p2',
    name: 'Jane Smith',
    age: 32,
    acuity_level: 4,
    arrival_time: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    status: 'waiting',
  },
  {
    id: 'p3',
    name: 'Bob Johnson',
    age: 67,
    acuity_level: 3,
    arrival_time: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    status: 'waiting',
  },
  {
    id: 'p4',
    name: 'Alice Brown',
    age: 28,
    acuity_level: 2,
    arrival_time: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
    status: 'waiting',
  },
  {
    id: 'p5',
    name: 'David Wilson',
    age: 55,
    acuity_level: 4,
    arrival_time: new Date(Date.now() - 10 * 60 * 1000).toISOString(),
    status: 'waiting',
  },
]

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

export const mockDischargeAnalysis = [
  {
    patient_id: 'p001',
    readiness_score: 85,
    estimated_discharge_date: new Date(Date.now() + 2 * 24 * 60 * 60 * 1000).toISOString(),
    explanation: 'Patient has met all discharge criteria. Vital signs stable, medications adjusted.',
    status: 'ready' as const,
  },
  {
    patient_id: 'p002',
    readiness_score: 65,
    estimated_discharge_date: new Date(Date.now() + 5 * 24 * 60 * 60 * 1000).toISOString(),
    explanation: 'Patient improving but requires additional monitoring. Review lab results.',
    status: 'needs_review' as const,
  },
  {
    patient_id: 'p003',
    readiness_score: 45,
    estimated_discharge_date: new Date(Date.now() + 10 * 24 * 60 * 60 * 1000).toISOString(),
    explanation: 'Patient requires continued treatment. Infection markers elevated.',
    status: 'not_ready' as const,
  },
  {
    patient_id: 'p004',
    readiness_score: 92,
    estimated_discharge_date: new Date(Date.now() + 1 * 24 * 60 * 60 * 1000).toISOString(),
    explanation: 'Excellent recovery. All parameters normal. Ready for discharge.',
    status: 'ready' as const,
  },
  {
    patient_id: 'p005',
    readiness_score: 58,
    estimated_discharge_date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
    explanation: 'Moderate progress. Needs continued observation.',
    status: 'needs_review' as const,
  },
  {
    patient_id: 'p006',
    readiness_score: 78,
    estimated_discharge_date: new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString(),
    explanation: 'Good recovery trajectory. Minor follow-up needed.',
    status: 'ready' as const,
  },
]

export const mockStaff = [
  {
    id: 's1',
    name: 'Dr. Sarah Williams',
    role: 'Physician',
    department: 'Emergency',
    availability: true,
  },
  {
    id: 's2',
    name: 'Nurse Mike Chen',
    role: 'Registered Nurse',
    department: 'ICU',
    availability: true,
  },
  {
    id: 's3',
    name: 'Dr. Emily Rodriguez',
    role: 'Surgeon',
    department: 'Surgery',
    availability: false,
  },
  {
    id: 's4',
    name: 'Dr. James Anderson',
    role: 'Cardiologist',
    department: 'Cardiology',
    availability: true,
  },
  {
    id: 's5',
    name: 'Nurse Lisa Park',
    role: 'Registered Nurse',
    department: 'Emergency',
    availability: true,
  },
  {
    id: 's6',
    name: 'Dr. Michael Thompson',
    role: 'Orthopedic Surgeon',
    department: 'Surgery',
    availability: true,
  },
]

// To use mocks, uncomment and modify the API endpoints to return mock data
// Example: In endpoints.ts, you can add a check for MOCK_MODE

