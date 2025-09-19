/**
 * Pre-grading Evaluation model for storing grading predictions and analysis
 */

const { DataTypes } = require('sequelize');
const { sequelize } = require('../config/database');

const PregradingEvaluation = sequelize.define('PregradingEvaluation', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true
  },
  
  // Card Information
  card_name: {
    type: DataTypes.STRING,
    allowNull: false
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
  
  // Analysis Results from GemCheck
  centering_score: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    validate: {
      min: 0,
      max: 100
    }
  },
  
  edges_score: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    validate: {
      min: 0,
      max: 100
    }
  },
  
  corners_score: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    validate: {
      min: 0,
      max: 100
    }
  },
  
  surface_score: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    validate: {
      min: 0,
      max: 100
    }
  },
  
  overall_score: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: false,
    validate: {
      min: 0,
      max: 100
    }
  },
  
  // Grade Predictions
  predicted_grade: {
    type: DataTypes.INTEGER,
    allowNull: false,
    validate: {
      min: 1,
      max: 10
    }
  },
  
  predicted_grade_label: {
    type: DataTypes.STRING,
    allowNull: false
  },
  
  confidence_score: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: false,
    validate: {
      min: 0,
      max: 100
    }
  },
  
  grade_range_low: {
    type: DataTypes.INTEGER,
    allowNull: true,
    validate: {
      min: 1,
      max: 10
    }
  },
  
  grade_range_high: {
    type: DataTypes.INTEGER,
    allowNull: true,
    validate: {
      min: 1,
      max: 10
    }
  },
  
  // Analysis Details
  analysis_factors: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Detailed analysis factors that influenced the grade prediction'
  },
  
  comparison_data: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Similar cards used for comparison analysis'
  },
  
  defects_detected: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Array of detected defects with severity and location'
  },
  
  // Population Analysis
  population_percentile: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    comment: 'Where this card falls in the population distribution'
  },
  
  rarity_factor: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    comment: 'How rare this grade would be for this card'
  },
  
  // Market Analysis
  estimated_value_low: {
    type: DataTypes.DECIMAL(10, 2),
    allowNull: true
  },
  
  estimated_value_high: {
    type: DataTypes.DECIMAL(10, 2),
    allowNull: true
  },
  
  market_data_source: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  market_analysis_date: {
    type: DataTypes.DATE,
    allowNull: true
  },
  
  // Images
  uploaded_images: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Array of uploaded image paths for analysis'
  },
  
  analysis_overlay_images: {
    type: DataTypes.JSONB,
    allowNull: true,
    comment: 'Generated overlay images showing detected issues'
  },
  
  // Processing Information
  analysis_method: {
    type: DataTypes.STRING,
    allowNull: false,
    defaultValue: 'gemcheck_cv'
  },
  
  processing_time_ms: {
    type: DataTypes.INTEGER,
    allowNull: true
  },
  
  ml_model_version: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  // Validation (if actual PSA grade is known)
  actual_psa_grade: {
    type: DataTypes.INTEGER,
    allowNull: true,
    validate: {
      min: 1,
      max: 10
    }
  },
  
  actual_cert_number: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  prediction_accuracy: {
    type: DataTypes.DECIMAL(5, 2),
    allowNull: true,
    comment: 'How accurate the prediction was if actual grade is known'
  },
  
  // User Information
  user_email: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  session_id: {
    type: DataTypes.STRING,
    allowNull: true
  },
  
  // Status
  is_public: {
    type: DataTypes.BOOLEAN,
    defaultValue: false
  },
  
  evaluation_notes: {
    type: DataTypes.TEXT,
    allowNull: true
  }
}, {
  tableName: 'pregrading_evaluations',
  indexes: [
    {
      fields: ['player_name']
    },
    {
      fields: ['year', 'brand']
    },
    {
      fields: ['predicted_grade']
    },
    {
      fields: ['overall_score']
    },
    {
      fields: ['created_at']
    },
    {
      fields: ['actual_psa_grade']
    }
  ]
});

// Instance methods
PregradingEvaluation.prototype.getGradeProbability = function() {
  const confidence = parseFloat(this.confidence_score);
  const grade = this.predicted_grade;
  
  // Calculate probability distribution around predicted grade
  const probabilities = {};
  for (let i = 1; i <= 10; i++) {
    if (i === grade) {
      probabilities[i] = confidence;
    } else {
      const distance = Math.abs(i - grade);
      probabilities[i] = Math.max(0, confidence - (distance * 15));
    }
  }
  
  return probabilities;
};

PregradingEvaluation.prototype.getRecommendations = function() {
  const recommendations = [];
  
  if (this.centering_score < 70) {
    recommendations.push({
      category: 'centering',
      severity: 'high',
      message: 'Poor centering detected - consider if grading cost is worth potential low grade'
    });
  }
  
  if (this.edges_score < 60) {
    recommendations.push({
      category: 'edges',
      severity: 'high',
      message: 'Significant edge wear detected - may limit maximum possible grade'
    });
  }
  
  if (this.corners_score < 70) {
    recommendations.push({
      category: 'corners',
      severity: 'medium',
      message: 'Corner wear detected - check for soft corners or damage'
    });
  }
  
  if (this.surface_score < 80) {
    recommendations.push({
      category: 'surface',
      severity: 'medium',
      message: 'Surface defects detected - examine for scratches or print lines'
    });
  }
  
  if (this.predicted_grade >= 9) {
    recommendations.push({
      category: 'grading',
      severity: 'positive',
      message: 'Card shows excellent potential - consider premium grading service'
    });
  }
  
  return recommendations;
};

// Class methods
PregradingEvaluation.getAccuracyStats = async function() {
  const { Op } = require('sequelize');
  
  const evaluations = await this.findAll({
    where: {
      actual_psa_grade: {
        [Op.not]: null
      },
      prediction_accuracy: {
        [Op.not]: null
      }
    },
    attributes: [
      [sequelize.fn('AVG', sequelize.col('prediction_accuracy')), 'avg_accuracy'],
      [sequelize.fn('COUNT', sequelize.col('id')), 'total_predictions'],
      [sequelize.fn('COUNT', sequelize.literal('CASE WHEN ABS(predicted_grade - actual_psa_grade) <= 1 THEN 1 END')), 'within_one_grade']
    ],
    raw: true
  });
  
  return evaluations[0];
};

PregradingEvaluation.getTrendAnalysis = async function(year, brand, playerName) {
  const { Op } = require('sequelize');
  
  return await this.findAll({
    where: {
      year,
      brand,
      player_name: {
        [Op.iLike]: `%${playerName}%`
      }
    },
    attributes: [
      'predicted_grade',
      [sequelize.fn('COUNT', sequelize.col('predicted_grade')), 'count'],
      [sequelize.fn('AVG', sequelize.col('overall_score')), 'avg_score']
    ],
    group: ['predicted_grade'],
    order: [['predicted_grade', 'DESC']],
    raw: true
  });
};

module.exports = PregradingEvaluation;