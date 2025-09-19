/**
 * PSA Card model for storing fetched card data
 */

const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const PsaCard = sequelize.define('PsaCard', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  
  // PSA Certificate Information
  cert_number: {
    type: DataTypes.STRING,
    allowNull: false,
    unique: true,
    index: true
  },
  
  grade: {
    type: DataTypes.INTEGER,
    allowNull: true,
    validate: {
      min: 1,
      max: 10
    }
  },
  
  grade_label: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  // Card Identification
  card_name: {
    type: DataTypes.STRING,
    allowNull: true
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
  
  set_name: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  card_number: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  variety: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  // Card Details
  sport: {
    type: DataTypes.STRING,
    allowNull: true,
    index: true
  },
  
  category: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  // Population Data
  total_population: {
    type: DataTypes.INTEGER,
    allowNull: true,
    defaultValue: 0
  },
  
  population_higher: {
    type: DataTypes.INTEGER,
    allowNull: true,
    defaultValue: 0
  },
  
  // Images
  front_image_url: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  back_image_url: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  label_image_url: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  // Raw API Response
  raw_api_data: {
    type: DataTypes.JSONB,
    allowNull: true
  },
  
  // Processing Status
  is_processed: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  },
  
  processing_error: {
    type: DataTypes.TEXT,
    allowNull: true
  },
  
  // Metadata
  date_graded: {
    type: DataTypes.DATE,
    allowNull: true
  },
  
  last_updated: {
    type: DataTypes.DATE,
    allowNull: false,
    defaultValue: DataTypes.NOW
  },
  
  api_fetch_date: {
    type: DataTypes.DATE,
    allowNull: false,
    defaultValue: DataTypes.NOW
  }
}, {
  tableName: 'psa_cards',
  indexes: [
    {
      fields: ['cert_number']
    },
    {
      fields: ['player_name']
    },
    {
      fields: ['year', 'brand']
    },
    {
      fields: ['sport', 'grade']
    },
    {
      fields: ['is_processed']
    }
  ]
});

// Instance methods
PsaCard.prototype.getGradeAnalysis = function() {
  return {
    grade: this.grade,
    gradeLabel: this.grade_label,
    population: {
      total: this.total_population,
      higher: this.population_higher,
      percentage: this.total_population > 0 ? 
        ((this.population_higher / this.total_population) * 100).toFixed(2) : 0
    }
  };
};

PsaCard.prototype.getCardIdentifier = function() {
  return `${this.year || ''} ${this.brand || ''} ${this.card_name || ''} #${this.card_number || ''}`.trim();
};

// Class methods
PsaCard.findBySimilarCards = async function(year, brand, playerName, limit = 10) {
  return await this.findAll({
    where: {
      year,
      brand,
      player_name: playerName,
      is_processed: true
    },
    order: [['grade', 'DESC'], ['date_graded', 'DESC']],
    limit
  });
};

PsaCard.getPopulationStats = async function(year, brand, cardName) {
  const { Op } = require('sequelize');
  
  return await this.findAll({
    where: {
      year,
      brand,
      card_name: {
        [Op.iLike]: `%${cardName}%`
      },
      is_processed: true
    },
    attributes: [
      'grade',
      [sequelize.fn('COUNT', sequelize.col('grade')), 'count']
    ],
    group: ['grade'],
    order: [['grade', 'DESC']],
    raw: true
  });
};

module.exports = PsaCard;