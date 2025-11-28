import dotenv from 'dotenv';
dotenv.config();

export const config = {
  port: process.env.PORT || 3000,
  nodeEnv: process.env.NODE_ENV || 'development',
  db: {
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'postgres',
    password: process.env.DB_PASS || 'postgres',
    name: process.env.DB_NAME || 'swasthya_db',
  },
  agents: {
    demandUrl: process.env.AGENT_DEMAND_URL || 'http://localhost:8001',
    staffUrl: process.env.AGENT_STAFF_URL || 'http://localhost:8002',
    erOrUrl: process.env.AGENT_EROR_URL || 'http://localhost:8003',
    dischargeUrl: process.env.AGENT_DISCHARGE_URL || 'http://localhost:8004',
    triageUrl: process.env.AGENT_TRIAGE_URL || 'http://localhost:8005',
  }
};
