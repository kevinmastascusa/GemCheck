/**
 * Winston logger configuration for PSA Pregrader
 */

const winston = require('winston');
const path = require('path');
require('dotenv').config();

const { LOG_LEVEL, LOG_FILE, NODE_ENV } = process.env;

// Create logs directory if it doesn't exist
const fs = require('fs');
const logsDir = path.dirname(LOG_FILE || 'logs/app.log');
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Custom format for logs
const logFormat = winston.format.combine(
  winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
  winston.format.errors({ stack: true }),
  winston.format.printf(({ level, message, timestamp, stack }) => {
    return `${timestamp} [${level.toUpperCase()}] ${stack || message}`;
  })
);

// Create logger instance
const logger = winston.createLogger({
  level: LOG_LEVEL || 'info',
  format: logFormat,
  transports: [
    // File transport
    new winston.transports.File({
      filename: path.join(logsDir, 'error.log'),
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 10
    }),
    new winston.transports.File({
      filename: LOG_FILE || 'logs/app.log',
      maxsize: 5242880, // 5MB
      maxFiles: 10
    })
  ]
});

// Add console transport in development
if (NODE_ENV !== 'production') {
  logger.add(new winston.transports.Console({
    format: winston.format.combine(
      winston.format.colorize(),
      winston.format.printf(({ level, message, timestamp }) => {
        return `${timestamp} [${level}] ${message}`;
      })
    )
  }));
}

// Create request logger middleware
const requestLogger = (req, res, next) => {
  const startTime = Date.now();
  
  // Log request
  logger.info(`📥 ${req.method} ${req.url} - IP: ${req.ip}`);
  
  // Log response when finished
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    const statusColor = res.statusCode >= 400 ? 'error' : 'info';
    logger[statusColor](`📤 ${req.method} ${req.url} - ${res.statusCode} - ${duration}ms`);
  });
  
  next();
};

// Error logging helper
const logError = (error, context = '') => {
  logger.error(`💥 ${context}: ${error.message}`, { 
    stack: error.stack,
    context 
  });
};

// API call logging helper
const logApiCall = (endpoint, status, duration) => {
  const level = status >= 400 ? 'warn' : 'info';
  logger[level](`🔌 API Call: ${endpoint} - Status: ${status} - Duration: ${duration}ms`);
};

// Database operation logging
const logDbOperation = (operation, table, duration) => {
  logger.debug(`🗃️  DB Operation: ${operation} on ${table} - ${duration}ms`);
};

module.exports = {
  logger,
  requestLogger,
  logError,
  logApiCall,
  logDbOperation
};