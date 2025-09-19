/**
 * Price comparison and market analysis routes
 */

const express = require('express');
const router = express.Router();
const { PriceComparison, PsaCard } = require('../models');
const { logger, logError } = require('../config/logger');

/**
 * Get market analysis for a card
 * GET /api/prices/market/:cardName
 */
router.get('/market/:cardName', async (req, res) => {
  try {
    const { cardName } = req.params;
    const { player, year, brand, grade } = req.query;
    
    const marketAnalysis = await PriceComparison.getMarketAnalysis(
      cardName,
      player,
      year ? parseInt(year) : null,
      brand
    );
    
    // Get recent sales
    const recentSales = await PriceComparison.findSimilarCards(
      cardName,
      player,
      year ? parseInt(year) : null,
      brand,
      grade ? parseInt(grade) : null,
      20
    );
    
    // Calculate trends
    const gradeMatrix = await PriceComparison.getGradeValueMatrix(
      cardName,
      player,
      year ? parseInt(year) : null,
      brand
    );
    
    res.json({
      success: true,
      data: {
        cardName,
        filters: { player, year, brand, grade },
        analysis: marketAnalysis,
        recentSales: recentSales.map(sale => ({
          id: sale.id,
          price: sale.sale_price,
          grade: sale.psa_grade,
          marketplace: sale.marketplace,
          saleDate: sale.sale_date,
          saleType: sale.sale_type,
          daysAgo: sale.getAgeInDays()
        })),
        gradeMatrix,
        summary: {
          totalSales: recentSales.length,
          priceRange: recentSales.length > 0 ? {
            min: Math.min(...recentSales.map(s => parseFloat(s.sale_price))),
            max: Math.max(...recentSales.map(s => parseFloat(s.sale_price)))
          } : null,
          lastUpdated: new Date().toISOString()
        }
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting market analysis');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get price history for a card
 * GET /api/prices/history
 */
router.get('/history', async (req, res) => {
  try {
    const { cardName, player, year, brand, grade, months = 12 } = req.query;
    
    if (!cardName) {
      return res.status(400).json({
        success: false,
        error: 'Card name is required'
      });
    }
    
    const priceHistory = await PriceComparison.getPriceHistory(
      cardName,
      player,
      year ? parseInt(year) : null,
      brand,
      grade ? parseInt(grade) : null,
      parseInt(months)
    );
    
    res.json({
      success: true,
      data: {
        cardName,
        grade: grade ? parseInt(grade) : 'all',
        period: `${months} months`,
        history: priceHistory,
        summary: {
          totalMonths: priceHistory.length,
          totalSales: priceHistory.reduce((sum, month) => sum + parseInt(month.sale_count), 0),
          overallTrend: priceHistory.length > 1 ? 
            (priceHistory[priceHistory.length - 1].avg_price > priceHistory[0].avg_price ? 'up' : 'down') : 'stable'
        }
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting price history');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Detect price outliers
 * GET /api/prices/outliers
 */
router.get('/outliers', async (req, res) => {
  try {
    const { cardName, player, year, brand, grade } = req.query;
    
    if (!cardName || !grade) {
      return res.status(400).json({
        success: false,
        error: 'Card name and grade are required for outlier detection'
      });
    }
    
    const outliers = await PriceComparison.detectOutliers(
      cardName,
      player,
      year ? parseInt(year) : null,
      brand,
      parseInt(grade)
    );
    
    res.json({
      success: true,
      data: {
        cardName,
        grade: parseInt(grade),
        outliers,
        summary: {
          totalOutliers: outliers.length,
          highPriceOutliers: outliers.filter(o => parseFloat(o.sale_price) > 0).length,
          detectionMethod: 'Standard deviation (2σ threshold)'
        }
      }
    });
    
  } catch (error) {
    logError(error, 'Error detecting price outliers');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Import price data (placeholder for future eBay/COMC integration)
 * POST /api/prices/import
 */
router.post('/import', async (req, res) => {
  try {
    const { source, cardName, priceData } = req.body;
    
    if (!source || !cardName || !Array.isArray(priceData)) {
      return res.status(400).json({
        success: false,
        error: 'Source, card name, and price data array are required'
      });
    }
    
    const importedPrices = [];
    const errors = [];
    
    for (const priceEntry of priceData) {
      try {
        const price = await PriceComparison.create({
          card_name: cardName,
          player_name: priceEntry.playerName,
          year: priceEntry.year,
          brand: priceEntry.brand,
          card_number: priceEntry.cardNumber,
          variety: priceEntry.variety,
          psa_grade: priceEntry.grade,
          cert_number: priceEntry.certNumber,
          sale_price: priceEntry.price,
          currency: priceEntry.currency || 'USD',
          sale_date: new Date(priceEntry.saleDate),
          marketplace: source,
          listing_title: priceEntry.title,
          listing_url: priceEntry.url,
          seller_rating: priceEntry.sellerRating,
          sale_type: priceEntry.saleType || 'auction',
          starting_bid: priceEntry.startingBid,
          bid_count: priceEntry.bidCount,
          data_source: source,
          source_item_id: priceEntry.itemId,
          is_verified: false
        });
        
        importedPrices.push(price);
      } catch (error) {
        errors.push({
          entry: priceEntry,
          error: error.message
        });
      }
    }
    
    logger.info(`💰 Imported ${importedPrices.length} price records from ${source}`);
    
    res.json({
      success: true,
      data: {
        imported: importedPrices.length,
        failed: errors.length,
        source,
        cardName,
        errors: errors.slice(0, 10) // Return first 10 errors
      }
    });
    
  } catch (error) {
    logError(error, 'Error importing price data');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

/**
 * Get price comparison for similar cards
 * GET /api/prices/compare
 */
router.get('/compare', async (req, res) => {
  try {
    const { cardName, player, year, brand, targetGrade } = req.query;
    
    if (!cardName || !targetGrade) {
      return res.status(400).json({
        success: false,
        error: 'Card name and target grade are required'
      });
    }
    
    const targetGradeInt = parseInt(targetGrade);
    const comparisons = {};
    
    // Get prices for the target grade and surrounding grades
    for (let grade = Math.max(1, targetGradeInt - 2); grade <= Math.min(10, targetGradeInt + 2); grade++) {
      const sales = await PriceComparison.findSimilarCards(
        cardName,
        player,
        year ? parseInt(year) : null,
        brand,
        grade,
        10
      );
      
      if (sales.length > 0) {
        const prices = sales.map(s => parseFloat(s.sale_price));
        comparisons[`grade_${grade}`] = {
          grade,
          saleCount: sales.length,
          avgPrice: (prices.reduce((a, b) => a + b, 0) / prices.length).toFixed(2),
          minPrice: Math.min(...prices),
          maxPrice: Math.max(...prices),
          recentSales: sales.slice(0, 5).map(s => ({
            price: s.sale_price,
            date: s.sale_date,
            marketplace: s.marketplace
          }))
        };
      }
    }
    
    res.json({
      success: true,
      data: {
        cardName,
        targetGrade: targetGradeInt,
        comparisons,
        summary: {
          gradesWithData: Object.keys(comparisons).length,
          totalSales: Object.values(comparisons).reduce((sum, g) => sum + g.saleCount, 0)
        }
      }
    });
    
  } catch (error) {
    logError(error, 'Error getting price comparison');
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;