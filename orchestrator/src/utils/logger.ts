import winston from 'winston';

export const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'orchestrator' },
  transports: [
    new winston.transports.Console({
      format: winston.format.simple(), // Human readable for dev
    }),
    // In a real deployment, we'd add File or HTTP transports here
  ],
});
