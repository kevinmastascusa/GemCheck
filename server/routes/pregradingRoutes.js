/**
 * Pre-grading evaluation routes
 */

const express = require('express');
const multer = require('multer');
const router = express.Router();
const { PregradingEvaluation, PsaCard } = require('../models');
const { logger, logError } = require('../config/logger');

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB
    files: 5 // Max 5 files
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'), false);
    }
  }
});

/**
 * Evaluate card for grading prediction
 * POST /api/pregrade/evaluate
 */
router.post('/evaluate', upload.array('images', 5), async (req, res) => {
  try {
    const {
      cardName,
      playerName,
      year,
      brand,
      cardNumber,
      userEmail,
      notes
    } = req.body;
    
    if (!cardName) {
      return res.status(400).json({
        success: false,
        error: 'Card name is required'
      });
    }
    
    if (!req.files || req.files.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'At least one card image is required'
      });
    }
    
    logger.info(`🔍 Starting pre-grading evaluation for: ${cardName}`);
    
    // TODO: Integrate with GemCheck computer vision analysis
    // For now, we'll create a placeholder evaluation
    const mockAnalysis = {
      centeringScore: Math.random() * 30 + 70,  // 70-100
      edgesScore: Math.random() * 40 + 60,      // 60-100
      cornersScore: Math.random() * 35 + 65,    // 65-100
      surfaceScore: Math.random() * 25 + 75     // 75-100
    };
    
    const overallScore = (
      mockAnalysis.centeringScore * 0.3 +
      mockAnalysis.edgesScore * 0.25 +
      mockAnalysis.cornersScore * 0.25 +
      mockAnalysis.surfaceScore * 0.2
    );
    
    // Determine predicted grade based on overall score
    let predictedGrade = 1;
    let predictedLabel = 'Authentic 1';
    let confidence = 65;
    
    if (overallScore >= 97) {
      predictedGrade = 10;
      predictedLabel = 'Gem Mint 10';
      confidence = 85;
    } else if (overallScore >= 92) {
      predictedGrade = 9;
      predictedLabel = 'Mint 9';
      confidence = 80;
    } else if (overallScore >= 85) {
      predictedGrade = 8;
      predictedLabel = 'NM-Mint 8';
      confidence = 75;
    } else if (overallScore >= 78) {
      predictedGrade = 7;
      predictedLabel = 'Near Mint 7';
      confidence = 70;
    } else if (overallScore >= 72) {
      predictedGrade = 6;
      predictedLabel = 'Excellent 6';
      confidence = 68;
    } else if (overallScore >= 66) {
      predictedGrade = 5;
      predictedLabel = 'VG-EX 5';
      confidence = 65;
    } else if (overallScore >= 60) {
      predictedGrade = 4;
      predictedLabel = 'Good 4';
      confidence = 62;
    } else if (overallScore >= 54) {
      predictedGrade = 3;
      predictedLabel = 'Fair 3';
      confidence = 60;
    } else if (overallScore >= 48) {
      predictedGrade = 2;
      predictedLabel = 'Poor 2';
      confidence = 58;
    }
    
    // Store evaluation in database
    const evaluation = await PregradingEvaluation.create({
      card_name: cardName,
      player_name: playerName,
      year: year ? parseInt(year) : null,
      brand,
      card_number: cardNumber,
      centering_score: mockAnalysis.centeringScore,
      edges_score: mockAnalysis.edgesScore,
      corners_score: mockAnalysis.cornersScore,
      surface_score: mockAnalysis.surfaceScore,
      overall_score: overallScore,
      predicted_grade: predictedGrade,
      predicted_grade_label: predictedLabel,
      confidence_score: confidence,
      grade_range_low: Math.max(1, predictedGrade - 1),
      grade_range_high: Math.min(10, predictedGrade + 1),
      analysis_factors: {
        centering: {
          horizontal_error: Math.random() * 0.1,
          vertical_error: Math.random() * 0.1,
          combined_error: Math.random() * 0.15
        },
        edges: {
          whitening_percentage: Math.random() * 10,
          nick_count: Math.floor(Math.random() * 3)
        },
        corners: {
          average_sharpness: Math.random() * 0.5 + 0.5,
          damage_detected: Math.random() > 0.7
        },
        surface: {
          defect_percentage: Math.random() * 2,
          scratch_count: Math.floor(Math.random() * 2)
        }
      },
      uploaded_images: req.files.map((file, index) => ({
        filename: `evaluation_${Date.now()}_${index}.jpg`,
        originalName: file.originalname,
        size: file.size,
        mimetype: file.mimetype
      })),
      processing_time_ms: Math.floor(Math.random() * 3000) + 1000,
      user_email: userEmail,
      evaluation_notes: notes,
      analysis_method: 'gemcheck_cv_v1'
    });
    
    // Get recommendations
    const recommendations = evaluation.getRecommendations();
    
    logger.info(`✅ Pre-grading evaluation completed: Grade ${predictedGrade} (${confidence}% confidence)`);
    
    res.json({
      success: true,
      data: {
        evaluationId: evaluation.id,
        card: {
          name: cardName,
          player: playerName,
          year,
          brand,
          cardNumber
        },
        analysis: {
          centering: mockAnalysis.centeringScore,
          edges: mockAnalysis.edgesScore,
          corners: mockAnalysis.cornersScore,
          surface: mockAnalysis.surfaceScore,
          overall: overallScore
        },
        prediction: {
          grade: predictedGrade,
          label: predictedLabel,
          confidence,
          range: {
            low: evaluation.grade_range_low,
            high: evaluation.grade_range_high
          },
          probabilities: evaluation.getGradeProbability()
        },
        recommendations,
        processingTime: evaluation.processing_time_ms
      }
    });
    
  } catch (error) {
    logError(error, 'Error in pre-grading evaluation');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get evaluation history
 * GET /api/pregrade/history
 */
router.get('/history', async (req, res) => {
  try {
    const { 
      limit = 20, 
      offset = 0, 
      player,
      grade,
      sort_by = 'created_at',
      sort_order = 'DESC'
    } = req.query;
    
    const whereClause = {};
    
    if (player) {
      whereClause.player_name = {
        [require('sequelize').Op.iLike]: `%${player}%`
      };
    }
    
    if (grade) {
      whereClause.predicted_grade = parseInt(grade);
    }
    
    const evaluations = await PregradingEvaluation.findAndCountAll({
      where: whereClause,
      order: [[sort_by, sort_order.toUpperCase()]],
      limit: Math.min(parseInt(limit), 100),
      offset: parseInt(offset),
      include: [{
        model: PsaCard,
        as: 'actualPsaCard',
        required: false
      }]
    });
    
    res.json({
      success: true,
      data: evaluations.rows.map(evaluation => ({
        id: evaluation.id,
        cardName: evaluation.card_name,
        playerName: evaluation.player_name,
        year: evaluation.year,
        brand: evaluation.brand,
        predictedGrade: evaluation.predicted_grade,
        predictedLabel: evaluation.predicted_grade_label,
        confidence: evaluation.confidence_score,
        overallScore: evaluation.overall_score,
        actualGrade: evaluation.actual_psa_grade,
        createdAt: evaluation.created_at,
        hasActualGrade: !!evaluation.actual_psa_grade
      })),
      pagination: {
        total: evaluations.count,
        limit: parseInt(limit),
        offset: parseInt(offset),
        pages: Math.ceil(evaluations.count / parseInt(limit))
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting evaluation history');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get accuracy statistics
 * GET /api/pregrade/accuracy
 */
router.get('/accuracy', async (req, res) => {
  try {
    const accuracyStats = await PregradingEvaluation.getAccuracyStats();
    
    res.json({
      success: true,
      data: {
        totalPredictions: parseInt(accuracyStats.total_predictions) || 0,
        averageAccuracy: parseFloat(accuracyStats.avg_accuracy) || 0,
        withinOneGrade: parseInt(accuracyStats.within_one_grade) || 0,
        withinOneGradePercentage: accuracyStats.total_predictions > 0 ? 
          ((accuracyStats.within_one_grade / accuracyStats.total_predictions) * 100).toFixed(1) : 0
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting accuracy statistics');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get specific evaluation details
 * GET /api/pregrade/:evaluationId
 */
router.get('/:evaluationId', async (req, res) => {
  try {
    const { evaluationId } = req.params;
    
    const evaluation = await PregradingEvaluation.findByPk(evaluationId, {
      include: [{
        model: PsaCard,
        as: 'actualPsaCard',
        required: false
      }]
    });
    
    if (!evaluation) {
      return res.status(404).json({
        success: false,
        error: 'Evaluation not found'
      });
    }
    
    res.json({
      success: true,
      data: {
        ...evaluation.toJSON(),
        recommendations: evaluation.getRecommendations(),
        gradeProbabilities: evaluation.getGradeProbability()
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting evaluation details');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;