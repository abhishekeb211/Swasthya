import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { config } from './config';
import { logger } from './utils/logger';
import { initScheduler } from './scheduler/jobs';
import apiRoutes from './routes/api';

const app = express();

// Middleware
app.use(helmet()); // Security headers
app.use(cors());
app.use(express.json());

// Routes
app.use('/api', apiRoutes);

// Start Server
app.listen(config.port, () => {
  logger.info(`Orchestrator running on port ${config.port}`);
  
  // Initialize Cron Jobs
  initScheduler();
});
