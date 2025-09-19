/**
 * Database configuration for PSA Pregrader
 */

const { Sequelize } = require('sequelize');
const logger = require('./logger');

// Load environment variables
require('dotenv').config();

const {
  DB_HOST,
  DB_PORT,
  DB_NAME,
  DB_USER,
  DB_PASSWORD,
  DB_SSL,
  NODE_ENV
} = process.env;

// Database configuration
const config = {
  host: DB_HOST || 'localhost',
  port: parseInt(DB_PORT) || 5432,
  dialect: 'postgres',
  logging: NODE_ENV === 'development' ? (msg) => logger.debug(msg) : false,
  dialectOptions: {
    ssl: DB_SSL === 'true' ? {
      require: true,
      rejectUnauthorized: false
    } : false
  },
  pool: {
    max: 10,
    min: 0,
    acquire: 30000,
    idle: 10000
  },
  define: {
    timestamps: true,
    underscored: true,
    freezeTableName: true
  }
};

// Create Sequelize instance
const sequelize = new Sequelize(DB_NAME, DB_USER, DB_PASSWORD, config);

// Test database connection
const testConnection = async () => {
  try {
    await sequelize.authenticate();
    logger.info('📊 Database connection established successfully');
    return true;
  } catch (error) {
    logger.error('❌ Unable to connect to database:', error.message);
    return false;
  }
};

// Initialize database
const initializeDatabase = async () => {
  try {
    const connected = await testConnection();
    if (!connected) {
      throw new Error('Database connection failed');
    }

    // Sync all models (create tables if they don't exist)
    await sequelize.sync({ 
      force: NODE_ENV === 'test',
      alter: NODE_ENV === 'development'
    });
    
    logger.info('🔄 Database models synchronized successfully');
    return true;
  } catch (error) {
    logger.error('💥 Database initialization failed:', error.message);
    throw error;
  }
};

// Graceful shutdown
const closeConnection = async () => {
  try {
    await sequelize.close();
    logger.info('📊 Database connection closed gracefully');
  } catch (error) {
    logger.error('❌ Error closing database connection:', error.message);
  }
};

module.exports = {
  sequelize,
  testConnection,
  initializeDatabase,
  closeConnection,
  Sequelize
};