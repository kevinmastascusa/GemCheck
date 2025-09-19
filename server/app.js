/**
 * Main Express application for PSA Pregrader
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { requestLogger, logger } = require('./config/logger');
const { initializeDatabase, closeConnection } = require('./config/database');

// Load environment variables
require('dotenv').config();

const {
  PORT = 3001,
  NODE_ENV = 'development',
  CORS_ORIGIN = 'http://localhost:3000,http://localhost:8501',
  RATE_LIMIT_WINDOW_MS = 900000,
  RATE_LIMIT_MAX_REQUESTS = 100
} = process.env;

class App {
  constructor() {
    this.app = express();
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }
  
  setupMiddleware() {
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
          connectSrc: ["'self'", "https://api.psacard.com"]
        }
      },
      crossOriginEmbedderPolicy: false
    }));
    
    // CORS configuration
    const corsOptions = {
      origin: CORS_ORIGIN.split(',').map(origin => origin.trim()),
      credentials: true,
      optionsSuccessStatus: 200,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
    };
    this.app.use(cors(corsOptions));
    
    // Request parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
    
    // Request logging
    this.app.use(requestLogger);
    
    // Rate limiting
    const limiter = rateLimit({
      windowMs: parseInt(RATE_LIMIT_WINDOW_MS),
      max: parseInt(RATE_LIMIT_MAX_REQUESTS),
      message: {
        success: false,
        error: 'Too many requests from this IP, please try again later.',
        retryAfter: Math.ceil(parseInt(RATE_LIMIT_WINDOW_MS) / 1000)
      },
      standardHeaders: true,
      legacyHeaders: false
    });
    
    this.app.use('/api/', limiter);
    
    // Health check endpoint (no rate limiting)
    this.app.get('/health', (req, res) => {
      res.json({
        success: true,
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        environment: NODE_ENV
      });
    });
  }
  
  setupRoutes() {
    // Import route handlers
    const psaRoutes = require('./routes/psaRoutes');
    const pregradingRoutes = require('./routes/pregradingRoutes');
    const usageRoutes = require('./routes/usageRoutes');
    const priceRoutes = require('./routes/priceRoutes');
    
    // Mount routes
    this.app.use('/api/psa', psaRoutes);
    this.app.use('/api/pregrade', pregradingRoutes);
    this.app.use('/api/usage', usageRoutes);
    this.app.use('/api/prices', priceRoutes);
    
    // Legacy route compatibility
    this.app.use('/api/card', psaRoutes);
    this.app.use('/api/cards', psaRoutes);
    this.app.use('/api/population', psaRoutes);
    
    // API documentation
    this.app.get('/api', (req, res) => {
      res.json({
        success: true,
        message: 'PSA Pregrader API v1.0',
        version: '1.0.0',
        documentation: {
          endpoints: {
            psa: {
              'POST /api/psa/fetch/:certNumber': 'Fetch card from PSA API',
              'GET /api/psa/card/:certNumber': 'Get stored card data',
              'GET /api/psa/search': 'Search cards by criteria',
              'POST /api/psa/batch-fetch': 'Batch fetch multiple cards',
              'GET /api/psa/stats': 'Get database statistics',
              'GET /api/psa/population/:certNumber': 'Population analysis'
            },
            pregrade: {
              'POST /api/pregrade/evaluate': 'Evaluate card for grading',
              'GET /api/pregrade/history': 'Get evaluation history',
              'GET /api/pregrade/accuracy': 'Get accuracy statistics'
            },
            usage: {
              'GET /api/usage/status': 'Check API usage status',
              'GET /api/usage/stats': 'Get usage statistics'
            },
            prices: {
              'GET /api/prices/market/:cardName': 'Get market analysis',
              'GET /api/prices/history': 'Get price history',
              'POST /api/prices/import': 'Import price data'
            }
          }
        },
        limits: {
          rateLimiting: `${RATE_LIMIT_MAX_REQUESTS} requests per ${RATE_LIMIT_WINDOW_MS / 1000} seconds`,
          psaApiDaily: '100 calls per day',
          maxBatchSize: '50 certificates'
        }
      });
    });
    
    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        success: false,
        error: 'Endpoint not found',
        path: req.originalUrl,
        method: req.method,
        suggestion: 'Check /api for available endpoints'
      });
    });
  }
  
  setupErrorHandling() {
    // Global error handler
    this.app.use((error, req, res, next) => {
      logger.error(`💥 Unhandled error: ${error.message}`, { 
        stack: error.stack,
        path: req.path,
        method: req.method
      });
      
      // Don't leak error details in production
      const isDev = NODE_ENV === 'development';
      
      res.status(error.status || 500).json({
        success: false,
        error: isDev ? error.message : 'Internal server error',
        ...(isDev && { stack: error.stack }),
        timestamp: new Date().toISOString()
      });
    });
    
    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('💥 Uncaught Exception:', error);
      process.exit(1);
    });
    
    // Handle unhandled rejections
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('💥 Unhandled Rejection at:', promise, 'reason:', reason);
    });
    
    // Graceful shutdown
    process.on('SIGINT', this.gracefulShutdown.bind(this));
    process.on('SIGTERM', this.gracefulShutdown.bind(this));
  }
  
  async gracefulShutdown(signal) {
    logger.info(`📴 Received ${signal}. Starting graceful shutdown...`);
    
    // Close server
    if (this.server) {
      this.server.close(() => {
        logger.info('🔌 HTTP server closed');
      });
    }
    
    // Close database connection
    await closeConnection();
    
    logger.info('👋 Graceful shutdown completed');
    process.exit(0);
  }
  
  async initialize() {
    try {
      // Initialize database
      await initializeDatabase();
      logger.info('🗄️  Database initialized successfully');
      
      return this.app;
    } catch (error) {
      logger.error('💥 Failed to initialize application:', error);
      throw error;
    }
  }
  
  async start() {
    try {
      await this.initialize();
      
      this.server = this.app.listen(PORT, () => {
        logger.info(`🚀 PSA Pregrader API server running on port ${PORT}`);
        logger.info(`🌍 Environment: ${NODE_ENV}`);
        logger.info(`📚 API Documentation: http://localhost:${PORT}/api`);
        logger.info(`❤️  Health Check: http://localhost:${PORT}/health`);
        
        if (NODE_ENV === 'development') {
          logger.info(`🔧 Development mode - detailed logging enabled`);
        }
      });
      
      return this.server;
    } catch (error) {
      logger.error('💥 Failed to start server:', error);
      throw error;
    }
  }
}

// Create and export app instance
const app = new App();

// Start server if this file is run directly
if (require.main === module) {
  app.start().catch(error => {
    logger.error('💥 Server startup failed:', error);
    process.exit(1);
  });
}

module.exports = app;