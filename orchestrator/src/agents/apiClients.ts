import axios from 'axios';
import { config } from '../config';
import { logger } from '../utils/logger';

// Generic interface for health check response
interface HealthStatus {
  service: string;
  status: 'healthy' | 'unhealthy';
  timestamp: string;
}

/**
 * Checks the health of a specific agent.
 * @param name - The human-readable name of the agent
 * @param url - The base URL of the agent service
 */
async function checkAgentHealth(name: string, url: string): Promise<HealthStatus> {
  try {
    await axios.get(`${url}/health`, { timeout: 2000 });
    return { service: name, status: 'healthy', timestamp: new Date().toISOString() };
  } catch (error) {
    logger.error(`Health check failed for ${name}`, { error: (error as Error).message });
    return { service: name, status: 'unhealthy', timestamp: new Date().toISOString() };
  }
}

/**
 * Aggregates health status from all 5 agents.
 */
export const getSystemHealth = async () => {
  const checks = [
    checkAgentHealth('Demand Forecast', config.agents.demandUrl),
    checkAgentHealth('Staff Scheduling', config.agents.staffUrl),
    checkAgentHealth('ER/OR Scheduling', config.agents.erOrUrl),
    checkAgentHealth('Discharge Planning', config.agents.dischargeUrl),
    checkAgentHealth('Triage Acuity', config.agents.triageUrl),
  ];
  return Promise.all(checks);
};

// --- Functional Wrappers for Agent Tasks ---

export const triggerDemandForecast = async () => {
  // Triggers the POST /predict endpoint on the Python agent
  const response = await axios.post(`${config.agents.demandUrl}/predict`);
  return response.data;
};

export const triggerStaffScheduling = async (forecastData: any) => {
  // Sends forecast data to the staff scheduler to optimize shifts
  const response = await axios.post(`${config.agents.staffUrl}/schedule`, { forecast: forecastData });
  return response.data;
};

export const triggerDischargeAnalysis = async () => {
  // Analyzes current inpatients for discharge readiness
  const response = await axios.post(`${config.agents.dischargeUrl}/analyze`);
  return response.data;
};
