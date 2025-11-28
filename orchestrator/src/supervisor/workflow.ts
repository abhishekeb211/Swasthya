import * as agents from '../agents/apiClients';
import { logger } from '../utils/logger';

/**
 * The Daily Hospital Readiness Routine.
 * 
 * Logic Flow:
 * 1. Forecast Demand: Predict patient volume for the next 24-48h.
 * 2. Optimize Staff: Adjust nurse/doctor shifts based on that volume.
 * 3. Plan Discharges: Identify beds that can be freed up.
 * 
 * This transforms reactive care into proactive operations.
 */
export const runDailyRoutine = async () => {
  const runId = new Date().toISOString().split('T')[0];
  logger.info(`Starting Daily Routine: ${runId}`);

  try {
    // Step 1: Forecast
    logger.info('Step 1: Running Demand Forecast...');
    const forecast = await agents.triggerDemandForecast();
    logger.info('Forecast generated successfully', { volume: forecast.predicted_volume });

    // Step 2: Scheduling
    // We pass the forecast to the scheduler. If demand is high, it adds staff.
    logger.info('Step 2: Optimizing Staff Schedules...');
    const schedule = await agents.triggerStaffScheduling(forecast);
    logger.info('Staff schedule optimized', { shifts_assigned: schedule.total_shifts });

    // Step 3: Discharge Planning
    // Free up capacity for the incoming demand.
    logger.info('Step 3: Analyzing Discharge Candidates...');
    const discharges = await agents.triggerDischargeAnalysis();
    logger.info('Discharge analysis complete', { candidates: discharges.count });

    logger.info(`Daily Routine ${runId} Completed Successfully.`);
    return {
      status: 'success',
      runId,
      summary: {
        forecast: forecast.predicted_volume,
        shifts: schedule.total_shifts,
        discharge_candidates: discharges.count
      }
    };

  } catch (error) {
    logger.error('Daily Routine Failed', { error });
    throw error; // Propagate to controller for error response
  }
};
