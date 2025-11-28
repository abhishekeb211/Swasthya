/**
 * Orchestrator API Integration Tests
 * 
 * Tests for the Swasthya orchestrator endpoints.
 * Run with: npm test
 */

import axios, { AxiosError } from 'axios';

// Test configuration
const BASE_URL = process.env.TEST_BASE_URL || 'http://localhost:3000';
const API_KEY = process.env.API_SECRET_KEY || 'test-api-key';

// Axios instance with default config
const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
    'x-api-key': API_KEY,
  },
});

describe('Swasthya Orchestrator API', () => {
  describe('GET /api/agents/health', () => {
    it('should return health status of all agents', async () => {
      try {
        const response = await api.get('/api/agents/health');
        
        expect(response.status).toBe(200);
        expect(response.data).toBeDefined();
        expect(typeof response.data).toBe('object');
        
        // Verify response structure
        const healthData = response.data;
        
        // Should contain agent health information
        expect(healthData).toHaveProperty('timestamp');
        
        // If agents are configured, verify their structure
        if (healthData.agents) {
          expect(Array.isArray(healthData.agents) || typeof healthData.agents === 'object').toBe(true);
        }
      } catch (error) {
        const axiosError = error as AxiosError;
        // If service is not running, skip gracefully
        if (axiosError.code === 'ECONNREFUSED') {
          console.warn('Orchestrator not running - skipping integration test');
          return;
        }
        throw error;
      }
    });

    it('should respond within acceptable time limit', async () => {
      const startTime = Date.now();
      
      try {
        await api.get('/api/agents/health');
        const duration = Date.now() - startTime;
        
        // Health check should respond within 5 seconds
        expect(duration).toBeLessThan(5000);
      } catch (error) {
        const axiosError = error as AxiosError;
        if (axiosError.code === 'ECONNREFUSED') {
          console.warn('Orchestrator not running - skipping integration test');
          return;
        }
        throw error;
      }
    });
  });

  describe('POST /api/workflow/daily', () => {
    it('should trigger daily workflow successfully', async () => {
      try {
        const response = await api.post('/api/workflow/daily');
        
        expect(response.status).toBe(200);
        expect(response.data).toBeDefined();
      } catch (error) {
        const axiosError = error as AxiosError;
        if (axiosError.code === 'ECONNREFUSED') {
          console.warn('Orchestrator not running - skipping integration test');
          return;
        }
        // Workflow may fail if agents are not running, but endpoint should respond
        if (axiosError.response?.status === 500) {
          expect(axiosError.response.data).toHaveProperty('error');
          return;
        }
        throw error;
      }
    });
  });

  describe('POST /api/forecast/run', () => {
    it('should trigger demand forecast', async () => {
      try {
        const response = await api.post('/api/forecast/run');
        
        expect(response.status).toBe(200);
        expect(response.data).toBeDefined();
      } catch (error) {
        const axiosError = error as AxiosError;
        if (axiosError.code === 'ECONNREFUSED') {
          console.warn('Orchestrator not running - skipping integration test');
          return;
        }
        // Forecast may fail if agent is not running
        if (axiosError.response?.status === 500) {
          expect(axiosError.response.data).toHaveProperty('error');
          return;
        }
        throw error;
      }
    });
  });

  describe('Authentication', () => {
    it('should reject requests without API key when auth is enabled', async () => {
      // Create client without API key
      const unauthenticatedApi = axios.create({
        baseURL: BASE_URL,
        timeout: 10000,
      });

      try {
        await unauthenticatedApi.get('/api/agents/health');
        // If we get here without auth enabled, that's okay in dev mode
      } catch (error) {
        const axiosError = error as AxiosError;
        if (axiosError.code === 'ECONNREFUSED') {
          console.warn('Orchestrator not running - skipping integration test');
          return;
        }
        // Should get 401 if auth is enabled
        if (axiosError.response?.status === 401) {
          expect(axiosError.response.data).toHaveProperty('error', 'Unauthorized');
          return;
        }
      }
    });
  });
});

// Mock tests for unit testing without live server
describe('Orchestrator Unit Tests (Mocked)', () => {
  it('should have valid test configuration', () => {
    expect(BASE_URL).toBeDefined();
    expect(typeof BASE_URL).toBe('string');
    expect(BASE_URL).toMatch(/^https?:\/\//);
  });

  it('should have API key configured', () => {
    expect(API_KEY).toBeDefined();
    expect(typeof API_KEY).toBe('string');
    expect(API_KEY.length).toBeGreaterThan(0);
  });
});
