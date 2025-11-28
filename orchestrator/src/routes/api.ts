import { Router, Request, Response } from 'express';
import { getSystemHealth, triggerDemandForecast } from '../agents/apiClients';
import { runDailyRoutine } from '../supervisor/workflow';

const router = Router();

// Consolidated Health Check
router.get('/agents/health', async (req: Request, res: Response) => {
  const health = await getSystemHealth();
  res.json(health);
});

// Manual Trigger for Daily Workflow
router.post('/workflow/daily', async (req: Request, res: Response) => {
  try {
    const result = await runDailyRoutine();
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Workflow failed', details: (error as Error).message });
  }
});

// Manual Trigger for Demand Forecast
router.post('/forecast/run', async (req: Request, res: Response) => {
  try {
    const result = await triggerDemandForecast();
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Forecast failed' });
  }
});

export default router;
