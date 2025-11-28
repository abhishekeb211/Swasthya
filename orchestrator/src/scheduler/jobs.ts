import cron from 'node-cron';
import { runDailyRoutine } from '../supervisor/workflow';
import { logger } from '../utils/logger';

export const initScheduler = () => {
  // Schedule: Runs at 00:00 (Midnight) every day
  cron.schedule('0 0 * * *', async () => {
    logger.info('Triggering scheduled daily workflow...');
    try {
      await runDailyRoutine();
    } catch (e) {
      logger.error('Scheduled workflow failed', e);
    }
  });
  
  logger.info('Cron jobs initialized: Daily Workflow @ Midnight');
};
