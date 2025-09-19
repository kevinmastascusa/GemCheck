/**
 * PSA Controller for handling card-related API endpoints
 */

const { PsaCard, PriceComparison, PregradingEvaluation } = require('../models');
const psaApiService = require('../services/psaApiService');
const { logger, logError } = require('../config/logger');
const { Op } = require('sequelize');

class PsaController {
  
  /**
   * Fetch and store card data from PSA API
   * POST /api/card/fetch/:certNumber
   */
  async fetchCard(req, res) {
    try {
      const { certNumber } = req.params;
      const { force = false } = req.query;
      
      // Validate cert number
      if (!certNumber || typeof certNumber !== 'string') {
        return res.status(400).json({
          success: false,
          error: 'Invalid certificate number provided'
        });
      }
      
      logger.info(`🔍 Fetching PSA card: ${certNumber} (force: ${force})`);
      
      // Check if card already exists and not forcing refresh
      if (!force) {
        const existingCard = await PsaCard.findOne({
          where: { cert_number: certNumber }
        });
        
        if (existingCard) {
          return res.json({
            success: true,
            data: existingCard,
            cached: true,
            message: 'Card data retrieved from database'
          });
        }
      }
      
      // Fetch from PSA API
      const apiResult = await psaApiService.getCardByCertNumber(certNumber);
      
      if (!apiResult.success) {
        return res.status(404).json({
          success: false,
          error: apiResult.error,
          certNumber
        });
      }
      
      // Store or update in database
      const [card, created] = await PsaCard.upsert({
        cert_number: certNumber,
        grade: apiResult.data.grade,
        grade_label: apiResult.data.gradeLabel,
        card_name: apiResult.data.cardName,
        player_name: apiResult.data.playerName,
        year: apiResult.data.year,
        brand: apiResult.data.brand,
        set_name: apiResult.data.setName,
        card_number: apiResult.data.cardNumber,
        variety: apiResult.data.variety,
        sport: apiResult.data.sport,
        category: apiResult.data.category,
        total_population: apiResult.data.totalPopulation,
        population_higher: apiResult.data.populationHigher,
        front_image_url: apiResult.data.frontImageUrl,
        back_image_url: apiResult.data.backImageUrl,
        label_image_url: apiResult.data.labelImageUrl,
        raw_api_data: apiResult.data.rawApiData,
        date_graded: apiResult.data.dateGraded,
        is_processed: true,
        last_updated: new Date(),
        api_fetch_date: new Date()
      }, {
        returning: true
      });
      
      logger.info(`✅ Card ${certNumber} ${created ? 'created' : 'updated'} successfully`);
      
      res.json({
        success: true,
        data: card,
        cached: false,
        created,
        responseTime: apiResult.responseTime
      });
      
    } catch (error) {
      logError(error, 'Error in fetchCard controller');
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Get stored card data by certificate number
   * GET /api/card/:certNumber
   */
  async getCard(req, res) {
    try {
      const { certNumber } = req.params;
      const { include_analysis = false, include_prices = false } = req.query;
      
      const includeOptions = [];
      
      if (include_analysis === 'true') {
        includeOptions.push({
          model: PregradingEvaluation,
          as: 'pregradingEvaluations',
          limit: 10,
          order: [['created_at', 'DESC']]
        });
      }
      
      if (include_prices === 'true') {
        includeOptions.push({
          model: PriceComparison,
          as: 'priceComparisons',
          limit: 20,
          order: [['sale_date', 'DESC']]
        });
      }
      
      const card = await PsaCard.findOne({
        where: { cert_number: certNumber },
        include: includeOptions
      });
      
      if (!card) {
        return res.status(404).json({
          success: false,
          error: 'Card not found in database',
          suggestion: `Try fetching it first: POST /api/card/fetch/${certNumber}`
        });
      }
      
      res.json({
        success: true,
        data: card
      });
      
    } catch (error) {
      logError(error, 'Error in getCard controller');
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Search stored cards by player, year, brand, etc.
   * GET /api/cards/search
   */
  async searchCards(req, res) {
    try {
      const { 
        player, 
        year, 
        brand, 
        sport,
        grade,
        grade_min,
        grade_max,
        limit = 20, 
        offset = 0,
        sort_by = 'created_at',
        sort_order = 'DESC'
      } = req.query;
      
      const whereClause = {
        is_processed: true
      };
      
      // Build search criteria
      if (player) {
        whereClause.player_name = {
          [Op.iLike]: `%${player}%`
        };
      }
      
      if (year) {
        whereClause.year = parseInt(year);
      }
      
      if (brand) {
        whereClause.brand = {
          [Op.iLike]: `%${brand}%`
        };
      }
      
      if (sport) {
        whereClause.sport = {
          [Op.iLike]: `%${sport}%`
        };
      }
      
      if (grade) {
        whereClause.grade = parseInt(grade);
      } else if (grade_min || grade_max) {
        whereClause.grade = {};
        if (grade_min) whereClause.grade[Op.gte] = parseInt(grade_min);
        if (grade_max) whereClause.grade[Op.lte] = parseInt(grade_max);
      }
      
      // Validate sort parameters
      const allowedSortFields = ['created_at', 'grade', 'year', 'player_name', 'brand'];
      const sortBy = allowedSortFields.includes(sort_by) ? sort_by : 'created_at';
      const sortOrder = ['ASC', 'DESC'].includes(sort_order.toUpperCase()) ? sort_order.toUpperCase() : 'DESC';
      
      const cards = await PsaCard.findAndCountAll({
        where: whereClause,
        order: [[sortBy, sortOrder]],
        limit: Math.min(parseInt(limit), 100), // Max 100 results
        offset: parseInt(offset)
      });
      
      res.json({
        success: true,
        data: cards.rows,
        pagination: {
          total: cards.count,
          limit: parseInt(limit),
          offset: parseInt(offset),
          pages: Math.ceil(cards.count / parseInt(limit))
        },
        filters: {
          player,
          year,
          brand,
          sport,
          grade,
          grade_min,
          grade_max
        }
      });
      
    } catch (error) {
      logError(error, 'Error in searchCards controller');
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Get population analysis for a specific card
   * GET /api/population/analysis/:certNumber
   */
  async getPopulationAnalysis(req, res) {
    try {
      const { certNumber } = req.params;
      
      const card = await PsaCard.findOne({
        where: { cert_number: certNumber }
      });
      
      if (!card) {
        return res.status(404).json({
          success: false,
          error: 'Card not found'
        });
      }
      
      // Get population stats for similar cards
      const populationStats = await PsaCard.getPopulationStats(
        card.year,
        card.brand,
        card.card_name
      );
      
      // Find similar cards
      const similarCards = await PsaCard.findBySimilarCards(
        card.year,
        card.brand,
        card.player_name,
        50
      );
      
      // Calculate percentile ranking
      const higherGrades = similarCards.filter(c => c.grade > card.grade).length;
      const percentile = similarCards.length > 0 ? 
        ((similarCards.length - higherGrades) / similarCards.length * 100).toFixed(1) : 0;
      
      res.json({
        success: true,
        data: {
          card: {
            certNumber: card.cert_number,
            grade: card.grade,
            cardName: card.card_name,
            playerName: card.player_name,
            year: card.year,
            brand: card.brand
          },
          population: {
            total: card.total_population,
            higher: card.population_higher,
            thisGrade: card.total_population - card.population_higher,
            percentile: parseFloat(percentile)
          },
          gradeDistribution: populationStats,
          similarCards: similarCards.map(c => ({
            certNumber: c.cert_number,
            grade: c.grade,
            cardName: c.card_name,
            dateGraded: c.date_graded
          }))
        }
      });
      
    } catch (error) {
      logError(error, 'Error in getPopulationAnalysis controller');
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Batch fetch multiple certificates
   * POST /api/cards/batch-fetch
   */
  async batchFetch(req, res) {
    try {
      const { certNumbers, options = {} } = req.body;
      
      if (!Array.isArray(certNumbers) || certNumbers.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'Invalid certificate numbers array'
        });
      }
      
      if (certNumbers.length > 50) {
        return res.status(400).json({
          success: false,
          error: 'Maximum 50 certificates per batch request'
        });
      }
      
      logger.info(`📦 Starting batch fetch for ${certNumbers.length} certificates`);
      
      const batchResult = await psaApiService.batchFetchCertificates(certNumbers, options);
      
      // Store successful results in database
      const storedCards = [];
      for (const result of batchResult.successful) {
        if (result.data) {
          try {
            const [card] = await PsaCard.upsert({
              cert_number: result.data.certNumber,
              grade: result.data.grade,
              grade_label: result.data.gradeLabel,
              card_name: result.data.cardName,
              player_name: result.data.playerName,
              year: result.data.year,
              brand: result.data.brand,
              set_name: result.data.setName,
              card_number: result.data.cardNumber,
              variety: result.data.variety,
              sport: result.data.sport,
              category: result.data.category,
              total_population: result.data.totalPopulation,
              population_higher: result.data.populationHigher,
              front_image_url: result.data.frontImageUrl,
              back_image_url: result.data.backImageUrl,
              label_image_url: result.data.labelImageUrl,
              raw_api_data: result.data.rawApiData,
              date_graded: result.data.dateGraded,
              is_processed: true,
              last_updated: new Date(),
              api_fetch_date: new Date()
            }, {
              returning: true
            });
            
            storedCards.push(card);
          } catch (dbError) {
            logError(dbError, `Error storing card ${result.data.certNumber}`);
          }
        }
      }
      
      res.json({
        success: true,
        data: {
          requested: batchResult.totalRequested,
          successful: batchResult.totalSuccessful,
          failed: batchResult.totalFailed,
          stored: storedCards.length,
          cards: storedCards,
          errors: batchResult.failed
        }
      });
      
    } catch (error) {
      logError(error, 'Error in batchFetch controller');
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
  
  /**
   * Get card statistics
   * GET /api/cards/stats
   */
  async getStats(req, res) {
    try {
      const stats = await PsaCard.findAll({
        attributes: [
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.col('id')), 'total_cards'],
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.literal('CASE WHEN grade = 10 THEN 1 END')), 'gem_mint_count'],
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.literal('CASE WHEN grade >= 9 THEN 1 END')), 'high_grade_count'],
          [PsaCard.sequelize.fn('AVG', PsaCard.sequelize.col('grade')), 'average_grade'],
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.literal('DISTINCT player_name')), 'unique_players'],
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.literal('DISTINCT brand')), 'unique_brands'],
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.literal('DISTINCT year')), 'unique_years']
        ],
        raw: true
      });
      
      // Get grade distribution
      const gradeDistribution = await PsaCard.findAll({
        attributes: [
          'grade',
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.col('grade')), 'count']
        ],
        where: {
          is_processed: true,
          grade: { [Op.not]: null }
        },
        group: ['grade'],
        order: [['grade', 'DESC']],
        raw: true
      });
      
      // Get top players by count
      const topPlayers = await PsaCard.findAll({
        attributes: [
          'player_name',
          [PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.col('player_name')), 'count'],
          [PsaCard.sequelize.fn('AVG', PsaCard.sequelize.col('grade')), 'avg_grade']
        ],
        where: {
          is_processed: true,
          player_name: { [Op.not]: null }
        },
        group: ['player_name'],
        order: [[PsaCard.sequelize.fn('COUNT', PsaCard.sequelize.col('player_name')), 'DESC']],
        limit: 10,
        raw: true
      });
      
      res.json({
        success: true,
        data: {
          overview: stats[0],
          gradeDistribution,
          topPlayers
        }
      });
      
    } catch (error) {
      logError(error, 'Error in getStats controller');
      res.status(500).json({
        success: false,
        error: error.message
      });
    }
  }
}

module.exports = new PsaController();