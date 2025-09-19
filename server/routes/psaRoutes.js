/**
 * PSA-related routes
 */

const express = require('express');
const router = express.Router();
const psaController = require('../controllers/psaController');

// Card fetching and storage routes
router.post('/fetch/:certNumber', psaController.fetchCard);
router.get('/card/:certNumber', psaController.getCard);
router.get('/search', psaController.searchCards);
router.post('/batch-fetch', psaController.batchFetch);

// Population and statistics routes
router.get('/population/:certNumber', psaController.getPopulationAnalysis);
router.get('/stats', psaController.getStats);

// Legacy compatibility routes
router.post('/card/fetch/:certNumber', psaController.fetchCard);
router.get('/cards/search', psaController.searchCards);
router.get('/population/analysis/:certNumber', psaController.getPopulationAnalysis);
router.post('/cards/batch-fetch', psaController.batchFetch);
router.get('/cards/stats', psaController.getStats);

module.exports = router;