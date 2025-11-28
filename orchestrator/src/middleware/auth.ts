/**
 * API Key Authentication Middleware
 * 
 * Validates incoming requests against the configured API secret key.
 * All protected routes must pass through this middleware.
 */

import { Request, Response, NextFunction } from 'express';
import { logger } from '../utils/logger';

const API_KEY_HEADER = 'x-api-key';

/**
 * Middleware to validate API key from request headers.
 * 
 * @param req - Express request object
 * @param res - Express response object  
 * @param next - Express next function
 * 
 * @returns 401 Unauthorized if key is missing or invalid
 */
export const validateApiKey = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const apiKey = req.header(API_KEY_HEADER);
  const validKey = process.env.API_SECRET_KEY;

  // Check if API key is configured
  if (!validKey) {
    logger.warn('API_SECRET_KEY not configured - authentication disabled');
    // In development, allow requests through if no key is configured
    if (process.env.NODE_ENV === 'development') {
      next();
      return;
    }
    res.status(500).json({
      error: 'Server Configuration Error',
      message: 'API authentication not configured',
    });
    return;
  }

  // Check if API key is provided in request
  if (!apiKey) {
    logger.warn(`Unauthorized request: Missing ${API_KEY_HEADER} header`, {
      ip: req.ip,
      path: req.path,
      method: req.method,
    });
    res.status(401).json({
      error: 'Unauthorized',
      message: `Missing required header: ${API_KEY_HEADER}`,
    });
    return;
  }

  // Validate API key (constant-time comparison to prevent timing attacks)
  if (!timingSafeEqual(apiKey, validKey)) {
    logger.warn('Unauthorized request: Invalid API key', {
      ip: req.ip,
      path: req.path,
      method: req.method,
    });
    res.status(401).json({
      error: 'Unauthorized',
      message: 'Invalid API key',
    });
    return;
  }

  // API key is valid, proceed to next middleware
  logger.debug('API key validated successfully', { path: req.path });
  next();
};

/**
 * Constant-time string comparison to prevent timing attacks.
 * 
 * @param a - First string
 * @param b - Second string
 * @returns true if strings are equal
 */
function timingSafeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  
  let result = 0;
  for (let i = 0; i < a.length; i++) {
    result |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return result === 0;
}

/**
 * Optional middleware to skip authentication for certain paths.
 * Use this for health check endpoints that need to be publicly accessible.
 */
export const publicRoute = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  next();
};

export default validateApiKey;
