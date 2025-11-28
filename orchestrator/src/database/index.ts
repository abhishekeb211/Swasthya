import { Pool } from 'pg';
import { config } from '../config';
import { logger } from '../utils/logger';

const pool = new Pool({
  host: config.db.host,
  user: config.db.user,
  password: config.db.password,
  database: config.db.name,
  port: 5432,
});

pool.on('error', (err) => {
  logger.error('Unexpected error on idle client', err);
  process.exit(-1);
});

// Helper for running queries
export const query = async (text: string, params?: any[]) => {
  const start = Date.now();
  const res = await pool.query(text, params);
  const duration = Date.now() - start;
  // Log slow queries for optimization
  if (duration > 1000) {
      logger.warn('Slow query executed', { text, duration, rows: res.rowCount });
  }
  return res;
};
