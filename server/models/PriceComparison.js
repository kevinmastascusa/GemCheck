/**
 * Price Comparison model for storing market price data for similar cards
 */

const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');
const moment = require('moment');

const PriceComparison = sequelize.define('PriceComparison', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  
  // Card Identification
  card_name: {
    type: DataTypes.STRING,
    allowNull: false,
    index: true
  },
  
  player_name: {
    type: DataTypes.STRING,
    allowNull: true,
    index: true
  },
  
  year: {
    type: DataTypes.INTEGER,
    allowNull: true,
    index: true
  },
  
  brand: {
    type: DataTypes.STRING,
    allowNull: true,
    index: true
  },
  
  card_number: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  variety: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  // Grade Information
  psa_grade: {
    type: DataTypes.INTEGER,
    allowNull: false,
    index: true,
    validate: {
      min: 1,
      max: 10
    }
  },
  
  cert_number: {
    type: DataTypes.STRING,
    allowNull: true,
    index: true
  },
  
  // Price Data
  sale_price: {
    type: DataTypes.DECIMAL(10, 2),
    allowNull: false
  },
  
  currency: {
    type: DataTypes.STRING,
    allowNull: false,
    defaultValue: 'USD'
  },
  
  sale_date: {
    type: DataTypes.DATE,
    allowNull: false,
    index: true
  },
  
  // Sale Details
  marketplace: {
    type: DataTypes.STRING,
    allowNull: false,
    index: true
  },
  
  listing_title: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  listing_url: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  seller_rating: {
    type: DataTypes.DECIMAL(3, 2),
    allowNull: true,
    validate: {
      min: 0,
      max: 5
    }
  },
  
  sale_type: {
    type: DataTypes.ENUM('auction', 'buy_it_now', 'best_offer', 'fixed_price'),
    allowNull: false,
    defaultValue: 'auction'
  },
  
  // Auction-specific data
  starting_bid: {
    type: DataTypes.DECIMAL(10, 2),
    allowNull: true
  },
  
  bid_count: {
    type: DataTypes.INTEGER,
    allowNull: true
  },
  
  reserve_met: {
    type: DataTypes.BOOLEAN,
    allowNull: true
  },
  
  // Market Analysis
  is_outlier: {
    type: DataTypes.BOOLEAN,
    allowNull: false,
    defaultValue: false
  },
  
  outlier_reason: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  condition_notes: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  // Data Source
  data_source: {
    type: DataTypes.STRING,
    allowNull: false
  },
  
  source_item_id: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  data_fetch_date: {
    type: DataTypes.DATE,
    allowNull: false,
    defaultValue: DataTypes.NOW
  },
  
  // Validation Status
  is_verified: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  },
  
  verification_notes: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  // Images
  sale_images: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Array of image URLs from the sale listing'
  }
}, {
  tableName: 'price_comparisons',
  indexes: [
    {
      fields: ['card_name', 'psa_grade']
    },
    {
      fields: ['player_name', 'year', 'brand']
    },
    {
      fields: ['marketplace', 'sale_date']
    },
    {
      fields: ['psa_grade', 'sale_price']
    },
    {
      fields: ['sale_date']
    }
  ]
});

// Instance methods
PriceComparison.prototype.getAgeInDays = function() {
  return moment().diff(moment(this.sale_date), 'days');
};

PriceComparison.prototype.isRecentSale = function(daysThreshold = 90) {
  return this.getAgeInDays() <= daysThreshold;
};

PriceComparison.prototype.getPricePerGrade = function() {
  return this.psa_grade > 0 ? (parseFloat(this.sale_price) / this.psa_grade).toFixed(2) : 0;
};

// Class methods
PriceComparison.findSimilarCards = async function(cardName, playerName, year, brand, grade, limit = 20) {
  const { Op } = require('sequelize');
  
  // Build similarity query with fuzzy matching
  const whereClause = {
    psa_grade: grade,
    sale_date: {
      [Op.gte]: moment().subtract(2, 'years').toDate()
    },
    is_verified: true
  };
  
  // Add card-specific filters
  if (cardName) {
    whereClause.card_name = {
      [Op.iLike]: `%${cardName}%`
    };
  }
  
  if (playerName) {
    whereClause.player_name = {
      [Op.iLike]: `%${playerName}%`
    };
  }
  
  if (year) {
    whereClause.year = year;
  }
  
  if (brand) {
    whereClause.brand = {
      [Op.iLike]: `%${brand}%`
    };
  }
  
  return await this.findAll({
    where: whereClause,
    order: [['sale_date', 'DESC']],
    limit
  });
};

PriceComparison.getMarketAnalysis = async function(cardName, playerName, year, brand) {
  const { Op } = require('sequelize');
  
  const whereClause = {
    sale_date: {
      [Op.gte]: moment().subtract(1, 'year').toDate()
    },
    is_verified: true
  };
  
  // Add filters
  if (cardName) whereClause.card_name = { [Op.iLike]: `%${cardName}%` };
  if (playerName) whereClause.player_name = { [Op.iLike]: `%${playerName}%` };
  if (year) whereClause.year = year;
  if (brand) whereClause.brand = { [Op.iLike]: `%${brand}%` };
  
  return await this.findAll({
    where: whereClause,
    attributes: [
      'psa_grade',
      [sequelize.fn('COUNT', sequelize.col('psa_grade')), 'sale_count'],
      [sequelize.fn('AVG', sequelize.col('sale_price')), 'avg_price'],
      [sequelize.fn('MIN', sequelize.col('sale_price')), 'min_price'],
      [sequelize.fn('MAX', sequelize.col('sale_price')), 'max_price'],
      [sequelize.fn('PERCENTILE_CONT', 0.5), sequelize.literal('WITHIN GROUP (ORDER BY sale_price)'), 'median_price']
    ],
    group: ['psa_grade'],
    order: [['psa_grade', 'DESC']],
    raw: true
  });
};

PriceComparison.getPriceHistory = async function(cardName, playerName, year, brand, grade, months = 12) {
  const { Op } = require('sequelize');
  
  const whereClause = {
    psa_grade: grade,
    sale_date: {
      [Op.gte]: moment().subtract(months, 'months').toDate()
    }
  };
  
  // Add filters
  if (cardName) whereClause.card_name = { [Op.iLike]: `%${cardName}%` };
  if (playerName) whereClause.player_name = { [Op.iLike]: `%${playerName}%` };
  if (year) whereClause.year = year;
  if (brand) whereClause.brand = { [Op.iLike]: `%${brand}%` };
  
  return await this.findAll({
    where: whereClause,
    attributes: [
      [sequelize.fn('DATE_TRUNC', 'month', sequelize.col('sale_date')), 'month'],
      [sequelize.fn('COUNT', sequelize.col('id')), 'sale_count'],
      [sequelize.fn('AVG', sequelize.col('sale_price')), 'avg_price'],
      [sequelize.fn('MIN', sequelize.col('sale_price')), 'min_price'],
      [sequelize.fn('MAX', sequelize.col('sale_price')), 'max_price']
    ],
    group: [sequelize.fn('DATE_TRUNC', 'month', sequelize.col('sale_date'))],
    order: [[sequelize.fn('DATE_TRUNC', 'month', sequelize.col('sale_date')), 'ASC']],
    raw: true
  });
};

PriceComparison.getGradeValueMatrix = async function(cardName, playerName, year, brand) {
  const { Op } = require('sequelize');
  
  const whereClause = {
    sale_date: {
      [Op.gte]: moment().subtract(6, 'months').toDate()
    },
    is_verified: true
  };
  
  // Add filters
  if (cardName) whereClause.card_name = { [Op.iLike]: `%${cardName}%` };
  if (playerName) whereClause.player_name = { [Op.iLike]: `%${playerName}%` };
  if (year) whereClause.year = year;
  if (brand) whereClause.brand = { [Op.iLike]: `%${brand}%` };
  
  const results = await this.findAll({
    where: whereClause,
    attributes: [
      'psa_grade',
      [sequelize.fn('AVG', sequelize.col('sale_price')), 'avg_price']
    ],
    group: ['psa_grade'],
    order: [['psa_grade', 'ASC']],
    raw: true
  });
  
  // Calculate grade multipliers
  const basePrice = results.find(r => r.psa_grade == 6)?.avg_price || results[0]?.avg_price || 0;
  
  return results.map(result => ({
    grade: result.psa_grade,
    avgPrice: parseFloat(result.avg_price || 0),
    multiplier: basePrice > 0 ? (parseFloat(result.avg_price) / basePrice).toFixed(2) : 1
  }));
};

PriceComparison.detectOutliers = async function(cardName, playerName, year, brand, grade) {
  const similar = await this.findSimilarCards(cardName, playerName, year, brand, grade, 100);
  
  if (similar.length < 10) return []; // Not enough data for outlier detection
  
  const prices = similar.map(card => parseFloat(card.sale_price));
  const mean = prices.reduce((a, b) => a + b) / prices.length;
  const stdDev = Math.sqrt(prices.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / prices.length);
  
  const outliers = [];
  const threshold = 2; // Standard deviations
  
  for (const card of similar) {
    const price = parseFloat(card.sale_price);
    const zScore = Math.abs((price - mean) / stdDev);
    
    if (zScore > threshold) {
      outliers.push({
        ...card.dataValues,
        z_score: zScore.toFixed(2),
        deviation_from_mean: ((price - mean) / mean * 100).toFixed(1) + '%'
      });
    }
  }
  
  return outliers;
};

module.exports = PriceComparison;