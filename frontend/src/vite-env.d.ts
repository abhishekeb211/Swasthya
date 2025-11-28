/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_ORCHESTRATOR_URL?: string
  readonly VITE_FORECAST_AGENT_URL?: string
  readonly VITE_STAFF_AGENT_URL?: string
  readonly VITE_ER_OR_AGENT_URL?: string
  readonly VITE_DISCHARGE_AGENT_URL?: string
  readonly VITE_TRIAGE_AGENT_URL?: string
  readonly VITE_FL_SERVER_1_URL?: string
  readonly VITE_FL_SERVER_2_URL?: string
  readonly VITE_MLFLOW_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

