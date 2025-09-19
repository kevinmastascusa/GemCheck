/**
 * Database setup script
 * Creates database tables and initial data
 */

const { initializeDatabase } = require('../config/database');
const { logger } = require('../config/logger');

async function setupDatabase() {
  try {
    logger.info('🗄️  Starting database setup...');
    
    // Initialize database and sync models
    await initializeDatabase();
    
    logger.info('✅ Database setup completed successfully!');
    logger.info('📊 Tables created:');
    logger.info('   - psa_cards (PSA certificate data)');
    logger.info('   - pregrading_evaluations (AI grading predictions)');
    logger.info('   - api_usage_tracking (API usage monitoring)');
    logger.info('   - price_comparisons (market price data)');
    
    process.exit(0);
  } catch (error) {
    logger.error('💥 Database setup failed:', error);
    process.exit(1);
  }
}

// Run setup if called directly
if (require.main === module) {
  setupDatabase();
}

module.exports = setupDatabase;