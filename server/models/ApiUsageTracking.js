/**
 * API Usage Tracking model for monitoring daily API call limits
 */

const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');
const moment = require('moment');

const ApiUsageTracking = sequelize.define('ApiUsageTracking', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  
  // Date tracking
  usage_date: {
    type: DataTypes.DATEONLY,
    allowNull: false,
    defaultValue: DataTypes.NOW,
    unique: true
  },
  
  // PSA API Usage
  psa_api_calls: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  
  psa_api_limit: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 100
  },
  
  psa_successful_calls: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  
  psa_failed_calls: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  
  // Other API Usage (eBay, COMC, etc.)
  ebay_api_calls: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  
  ebay_api_limit: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 1000
  },
  
  comc_api_calls: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  
  comc_api_limit: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 500
  },
  
  // Rate limiting info
  rate_limit_hits: {
    type: DataTypes.INTEGER,
    allowNull: false,
    defaultValue: 0
  },
  
  last_rate_limit_time: {
    type: DataTypes.DATE,
    allowNull: true
  },
  
  // API Response times (milliseconds)
  psa_avg_response_time: {
    type: DataTypes.DECIMAL(8, 2),
    allowNull: true
  },
  
  psa_max_response_time: {
    type: DataTypes.INTEGER,
    allowNull: true
  },
  
  // Error tracking
  error_details: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Array of error details with timestamps'
  },
  
  // Usage patterns
  peak_usage_hour: {
    type: DataTypes.INTEGER,
    allowNull: true,
    validate: {
      min: 0,
      max: 23
    }
  },
  
  hourly_distribution: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Hour-by-hour breakdown of API calls'
  }
}, {
  tableName: 'api_usage_tracking',
  indexes: [
    {
      fields: ['usage_date']
    },
    {
      unique: true,
      fields: ['usage_date']
    }
  ]
});

// Instance methods
ApiUsageTracking.prototype.canMakePsaCall = function() {
  return this.psa_api_calls < this.psa_api_limit;
};

ApiUsageTracking.prototype.getRemainingPsaCalls = function() {
  return Math.max(0, this.psa_api_limit - this.psa_api_calls);
};

ApiUsageTracking.prototype.getUsagePercentage = function(apiType = 'psa') {
  const calls = this[`${apiType}_api_calls`];
  const limit = this[`${apiType}_api_limit`];
  return limit > 0 ? (calls / limit * 100).toFixed(2) : 0;
};

ApiUsageTracking.prototype.addApiCall = async function(apiType, success = true, responseTime = null) {
  const callField = `${apiType}_api_calls`;
  const successField = `${apiType}_successful_calls`;
  const failedField = `${apiType}_failed_calls`;
  
  // Increment counters
  this[callField] = (this[callField] || 0) + 1;
  
  if (success) {
    this[successField] = (this[successField] || 0) + 1;
  } else {
    this[failedField] = (this[failedField] || 0) + 1;
  }
  
  // Update response time tracking for PSA API
  if (apiType === 'psa' && responseTime && success) {
    if (!this.psa_avg_response_time) {
      this.psa_avg_response_time = responseTime;
    } else {
      const currentAvg = parseFloat(this.psa_avg_response_time);
      const totalSuccessful = this.psa_successful_calls;
      this.psa_avg_response_time = ((currentAvg * (totalSuccessful - 1)) + responseTime) / totalSuccessful;
    }
    
    if (!this.psa_max_response_time || responseTime > this.psa_max_response_time) {
      this.psa_max_response_time = responseTime;
    }
  }
  
  // Update hourly distribution
  const currentHour = moment().hour();
  const hourlyDist = this.hourly_distribution || {};
  hourlyDist[currentHour] = (hourlyDist[currentHour] || 0) + 1;
  this.hourly_distribution = hourlyDist;
  
  // Find peak usage hour
  let maxCalls = 0;
  let peakHour = 0;
  for (const [hour, calls] of Object.entries(hourlyDist)) {
    if (calls > maxCalls) {
      maxCalls = calls;
      peakHour = parseInt(hour);
    }
  }
  this.peak_usage_hour = peakHour;
  
  await this.save();
  return this;
};

ApiUsageTracking.prototype.addError = async function(apiType, error) {
  const errorDetails = this.error_details || [];
  errorDetails.push({
    api_type: apiType,
    error_message: error.message,
    error_code: error.code || error.status,
    timestamp: new Date().toISOString()
  });
  
  // Keep only last 50 errors
  if (errorDetails.length > 50) {
    errorDetails.splice(0, errorDetails.length - 50);
  }
  
  this.error_details = errorDetails;
  await this.save();
  return this;
};

// Class methods
ApiUsageTracking.getTodayUsage = async function() {
  const today = moment().format('YYYY-MM-DD');
  
  let usage = await this.findOne({
    where: { usage_date: today }
  });
  
  if (!usage) {
    usage = await this.create({
      usage_date: today
    });
  }
  
  return usage;
};

ApiUsageTracking.getWeeklyStats = async function() {
  const { Op } = require('sequelize');
  const weekAgo = moment().subtract(7, 'days').format('YYYY-MM-DD');
  
  return await this.findAll({
    where: {
      usage_date: {
        [Op.gte]: weekAgo
      }
    },
    order: [['usage_date', 'DESC']],
    raw: true
  });
};

ApiUsageTracking.getMonthlyStats = async function() {
  const { Op } = require('sequelize');
  const monthAgo = moment().subtract(30, 'days').format('YYYY-MM-DD');
  
  const stats = await this.findAll({
    where: {
      usage_date: {
        [Op.gte]: monthAgo
      }
    },
    attributes: [
      [sequelize.fn('SUM', sequelize.col('psa_api_calls')), 'total_psa_calls'],
      [sequelize.fn('SUM', sequelize.col('psa_successful_calls')), 'total_successful'],
      [sequelize.fn('SUM', sequelize.col('psa_failed_calls')), 'total_failed'],
      [sequelize.fn('AVG', sequelize.col('psa_avg_response_time')), 'avg_response_time'],
      [sequelize.fn('MAX', sequelize.col('psa_max_response_time')), 'max_response_time'],
      [sequelize.fn('COUNT', sequelize.col('usage_date')), 'active_days']
    ],
    raw: true
  });
  
  return stats[0];
};

// Rate limiting helper
ApiUsageTracking.recordRateLimit = async function(apiType = 'psa') {
  const usage = await this.getTodayUsage();
  usage.rate_limit_hits += 1;
  usage.last_rate_limit_time = new Date();
  await usage.save();
  return usage;
};

module.exports = ApiUsageTracking;