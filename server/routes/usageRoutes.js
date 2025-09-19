/**
 * API Usage tracking routes
 */

const express = require('express');
const router = express.Router();
const psaApiService = require('../services/psaApiService');
const { ApiUsageTracking } = require('../models');
const { logger, logError } = require('../config/logger');

/**
 * Get current API usage status
 * GET /api/usage/status
 */
router.get('/status', async (req, res) => {
  try {
    const stats = await psaApiService.getUsageStats();
    
    res.json({
      success: true,
      data: {
        today: {
          ...stats.today,
          status: stats.today.remaining > 0 ? 'active' : 'limit_reached',
          resetTime: '00:00 UTC'
        },
        limits: {
          psaApi: 100,
          rateLimitWindow: '15 minutes',
          maxBatchSize: 50
        }
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting usage status');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get detailed usage statistics
 * GET /api/usage/stats
 */
router.get('/stats', async (req, res) => {
  try {
    const { period = 'week' } = req.query;
    
    let stats;
    switch (period) {
      case 'week':
        stats = await ApiUsageTracking.getWeeklyStats();
        break;
      case 'month':
        stats = await ApiUsageTracking.getMonthlyStats();
        break;
      case 'today':
        stats = [await ApiUsageTracking.getTodayUsage()];
        break;
      default:
        stats = await ApiUsageTracking.getWeeklyStats();
    }
    
    res.json({
      success: true,
      data: {
        period,
        stats,
        summary: period === 'month' ? await ApiUsageTracking.getMonthlyStats() : null
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting usage statistics');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Reset daily usage (admin only - for testing)
 * POST /api/usage/reset
 */
router.post('/reset', async (req, res) => {
  try {
    // In production, this should be protected with admin authentication
    if (process.env.NODE_ENV === 'production') {
      return res.status(403).json({
        success: false,
        error: 'Reset not allowed in production'
      });
    }
    
    const today = await ApiUsageTracking.getTodayUsage();
    await today.update({
      psa_api_calls: 0,
      psa_successful_calls: 0,
      psa_failed_calls: 0,
      rate_limit_hits: 0,
      last_rate_limit_time: null,
      error_details: []
    });
    
    logger.info('🔄 Daily usage statistics reset');
    
    res.json({
      success: true,
      message: 'Usage statistics reset successfully'
    });
    
  } catch (error) {
    logError(error, 'Error resetting usage statistics');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;