/**
 * Database models index file
 * Exports all models and defines their relationships
 */

const PsaCard = require('./PsaCard');
const PregradingEvaluation = require('./PregradingEvaluation');
const ApiUsageTracking = require('./ApiUsageTracking');
const PriceComparison = require('./PriceComparison');

// Define model relationships

// PSA Card can have many price comparisons
PsaCard.hasMany(PriceComparison, {
  foreignKey: 'cert_number',
  sourceKey: 'cert_number',
  as: 'priceComparisons'
});

PriceComparison.belongsTo(PsaCard, {
  foreignKey: 'cert_number',
  targetKey: 'cert_number',
  as: 'psaCard'
});

// PSA Card can be referenced by pre-grading evaluations for validation
PsaCard.hasMany(PregradingEvaluation, {
  foreignKey: 'actual_cert_number',
  sourceKey: 'cert_number',
  as: 'pregradingEvaluations'
});

PregradingEvaluation.belongsTo(PsaCard, {
  foreignKey: 'actual_cert_number',
  targetKey: 'cert_number',
  as: 'actualPsaCard'
});

// Pre-grading evaluations can have related price comparisons
PregradingEvaluation.hasMany(PriceComparison, {
  foreignKey: {
    name: 'related_evaluation_id',
    allowNull: true
  },
  as: 'relatedPrices'
});

PriceComparison.belongsTo(PregradingEvaluation, {
  foreignKey: {
    name: 'related_evaluation_id',
    allowNull: true
  },
  as: 'evaluation'
});

// Export all models
module.exports = {
  PsaCard,
  PregradingEvaluation,
  ApiUsageTracking,
  PriceComparison
};

// Export individual models for direct import
module.exports.PsaCard = PsaCard;
module.exports.PregradingEvaluation = PregradingEvaluation;
module.exports.ApiUsageTracking = ApiUsageTracking;
module.exports.PriceComparison = PriceComparison;