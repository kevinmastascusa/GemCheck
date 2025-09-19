/**
 * PSA API Service for fetching card data
 * Handles authentication, rate limiting, and data processing
 */

const axios = require('axios');
const { logger, logApiCall, logError } = require('../config/logger');
const { ApiUsageTracking } = require('../models');

require('dotenv').config();

class PsaApiService {
  constructor() {
    this.baseUrl = process.env.PSA_API_BASE_URL || 'https://api.psacard.com/publicapi/';
    this.apiToken = process.env.PSA_API_TOKEN;
    this.dailyLimit = parseInt(process.env.PSA_DAILY_LIMIT) || 100;
    
    if (!this.apiToken) {
      logger.warn('⚠️  PSA API token not configured - API calls will fail');
    }
    
    // Configure axios instance
    this.client = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Authorization': `bearer ${this.apiToken}`,
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
    
    // Add response interceptor for logging
    this.client.interceptors.response.use(
      (response) => {
        logApiCall(response.config.url, response.status, response.config.metadata?.duration || 0);
        return response;
      },
      (error) => {
        const status = error.response?.status || 0;
        const duration = error.config?.metadata?.duration || 0;
        logApiCall(error.config?.url || 'unknown', status, duration);
        return Promise.reject(error);
      }
    );
    
    // Add request interceptor for timing
    this.client.interceptors.request.use((config) => {
      config.metadata = { startTime: Date.now() };
      return config;
    });
    
    this.client.interceptors.response.use(
      (response) => {
        response.config.metadata.duration = Date.now() - response.config.metadata.startTime;
        return response;
      },
      (error) => {
        if (error.config) {
          error.config.metadata.duration = Date.now() - error.config.metadata.startTime;
        }
        return Promise.reject(error);
      }
    );
  }
  
  /**
   * Check if we can make an API call within daily limits
   */
  async canMakeApiCall() {
    try {
      const usage = await ApiUsageTracking.getTodayUsage();
      return usage.canMakePsaCall();
    } catch (error) {
      logError(error, 'Error checking API usage limits');
      return false;
    }
  }
  
  /**
   * Get remaining API calls for today
   */
  async getRemainingCalls() {
    try {
      const usage = await ApiUsageTracking.getTodayUsage();
      return usage.getRemainingPsaCalls();
    } catch (error) {
      logError(error, 'Error getting remaining API calls');
      return 0;
    }
  }
  
  /**
   * Fetch card data by certificate number
   */
  async getCardByCertNumber(certNumber) {
    const startTime = Date.now();
    
    try {
      // Validate cert number
      if (!certNumber || typeof certNumber !== 'string') {
        throw new Error('Invalid certificate number provided');
      }
      
      // Check API limits
      if (!(await this.canMakeApiCall())) {
        const remaining = await this.getRemainingCalls();
        throw new Error(`Daily API limit reached. ${remaining} calls remaining.`);
      }
      
      logger.info(`🔍 Fetching PSA card data for cert number: ${certNumber}`);
      
      // Make API call
      const response = await this.client.get(`cert/GetByCertNumber/${certNumber}`);
      const duration = Date.now() - startTime;
      
      // Record successful API usage
      const usage = await ApiUsageTracking.getTodayUsage();
      await usage.addApiCall('psa', true, duration);
      
      // Validate response
      if (!response.data) {
        throw new Error('Empty response from PSA API');
      }
      
      // Process and normalize the response data
      const normalizedData = this.normalizeCardData(response.data, certNumber);
      
      logger.info(`✅ Successfully fetched PSA card data for cert ${certNumber} in ${duration}ms`);
      return {
        success: true,
        data: normalizedData,
        rawData: response.data,
        responseTime: duration
      };
      
    } catch (error) {
      const duration = Date.now() - startTime;
      
      // Record failed API usage
      try {
        const usage = await ApiUsageTracking.getTodayUsage();
        await usage.addApiCall('psa', false, duration);
        await usage.addError('psa', error);
      } catch (usageError) {
        logError(usageError, 'Error recording API usage');
      }
      
      // Handle specific error types
      if (error.response) {
        const { status, data } = error.response;
        
        switch (status) {
          case 401:
            logError(error, 'PSA API authentication failed');
            throw new Error('PSA API authentication failed - check API token');
          
          case 404:
            logger.warn(`🔍 Certificate number ${certNumber} not found in PSA database`);
            return {
              success: false,
              error: 'Certificate number not found',
              certNumber,
              responseTime: duration
            };
          
          case 429:
            logError(error, 'PSA API rate limit exceeded');
            await ApiUsageTracking.recordRateLimit('psa');
            throw new Error('PSA API rate limit exceeded - please try again later');
          
          default:
            logError(error, `PSA API error (${status})`);
            throw new Error(`PSA API error: ${data?.message || error.message}`);
        }
      } else if (error.code === 'ENOTFOUND' || error.code === 'ECONNREFUSED') {
        logError(error, 'PSA API connection failed');
        throw new Error('Unable to connect to PSA API - check network connection');
      } else {
        logError(error, 'PSA API request failed');
        throw new Error(`PSA API request failed: ${error.message}`);
      }
    }
  }
  
  /**
   * Normalize PSA API response data to our internal format
   */
  normalizeCardData(apiData, certNumber) {
    try {
      // The exact structure depends on PSA API response format
      // This is a template based on common card grading API patterns
      return {
        certNumber: certNumber,
        grade: apiData.Grade || apiData.grade || null,
        gradeLabel: apiData.GradeLabel || apiData.grade_label || null,
        cardName: apiData.CardName || apiData.card_name || apiData.name || null,
        playerName: this.extractPlayerName(apiData.CardName || apiData.card_name),
        year: apiData.Year || apiData.year || this.extractYear(apiData.CardName || apiData.card_name),
        brand: apiData.Brand || apiData.brand || apiData.manufacturer || null,
        setName: apiData.SetName || apiData.set_name || apiData.set || null,
        cardNumber: apiData.CardNumber || apiData.card_number || apiData.number || null,
        variety: apiData.Variety || apiData.variety || null,
        sport: apiData.Sport || apiData.sport || apiData.category || null,
        category: apiData.Category || apiData.category || null,
        totalPopulation: apiData.TotalPopulation || apiData.total_population || 0,
        populationHigher: apiData.PopulationHigher || apiData.population_higher || 0,
        frontImageUrl: apiData.FrontImageUrl || apiData.front_image || apiData.image_front || null,
        backImageUrl: apiData.BackImageUrl || apiData.back_image || apiData.image_back || null,
        labelImageUrl: apiData.LabelImageUrl || apiData.label_image || apiData.image_label || null,
        dateGraded: this.parseDate(apiData.DateGraded || apiData.date_graded || apiData.graded_date),
        // Store the complete raw response for reference
        rawApiData: apiData
      };
    } catch (error) {
      logError(error, 'Error normalizing PSA card data');
      return {
        certNumber,
        rawApiData: apiData,
        processingError: error.message
      };
    }
  }
  
  /**
   * Extract player name from card name
   */
  extractPlayerName(cardName) {
    if (!cardName) return null;
    
    // Common patterns for player names in card titles
    const patterns = [
      /^([A-Za-z\s.'-]+?)(?:\s#|\s\d{4}|\s-|\s\()/,  // Name before # or year or dash
      /^([A-Za-z\s.'-]+)/  // Just the first part if no special characters
    ];
    
    for (const pattern of patterns) {
      const match = cardName.match(pattern);
      if (match) {
        return match[1].trim();
      }
    }
    
    return null;
  }
  
  /**
   * Extract year from card name
   */
  extractYear(cardName) {
    if (!cardName) return null;
    
    const yearMatch = cardName.match(/\b(19\d{2}|20\d{2})\b/);
    return yearMatch ? parseInt(yearMatch[1]) : null;
  }
  
  /**
   * Parse date string to Date object
   */
  parseDate(dateString) {
    if (!dateString) return null;
    
    try {
      const parsed = new Date(dateString);
      return isNaN(parsed.getTime()) ? null : parsed;
    } catch (error) {
      return null;
    }
  }
  
  /**
   * Batch fetch multiple certificates
   */
  async batchFetchCertificates(certNumbers, options = {}) {
    const { maxConcurrent = 3, delayBetweenRequests = 1000 } = options;
    
    if (!Array.isArray(certNumbers) || certNumbers.length === 0) {
      throw new Error('Invalid certificate numbers array provided');
    }
    
    // Check if we have enough API calls remaining
    const remainingCalls = await this.getRemainingCalls();
    if (remainingCalls < certNumbers.length) {
      throw new Error(`Insufficient API calls remaining: ${remainingCalls} available, ${certNumbers.length} requested`);
    }
    
    logger.info(`📦 Starting batch fetch for ${certNumbers.length} certificates`);
    
    const results = [];
    const errors = [];
    
    // Process in batches to avoid overwhelming the API
    for (let i = 0; i < certNumbers.length; i += maxConcurrent) {
      const batch = certNumbers.slice(i, i + maxConcurrent);
      
      const batchPromises = batch.map(async (certNumber) => {
        try {
          // Add delay between requests
          if (i > 0) {
            await new Promise(resolve => setTimeout(resolve, delayBetweenRequests));
          }
          
          const result = await this.getCardByCertNumber(certNumber);
          results.push(result);
          return result;
        } catch (error) {
          const errorResult = {
            success: false,
            certNumber,
            error: error.message
          };
          errors.push(errorResult);
          return errorResult;
        }
      });
      
      await Promise.all(batchPromises);
      
      // Log progress
      logger.info(`📦 Batch progress: ${Math.min(i + maxConcurrent, certNumbers.length)}/${certNumbers.length} completed`);
    }
    
    logger.info(`📦 Batch fetch completed: ${results.filter(r => r.success).length} successful, ${errors.length} failed`);
    
    return {
      successful: results.filter(r => r.success),
      failed: errors,
      totalRequested: certNumbers.length,
      totalSuccessful: results.filter(r => r.success).length,
      totalFailed: errors.length
    };
  }
  
  /**
   * Get API usage statistics
   */
  async getUsageStats() {
    try {
      const todayUsage = await ApiUsageTracking.getTodayUsage();
      const weeklyStats = await ApiUsageTracking.getWeeklyStats();
      const monthlyStats = await ApiUsageTracking.getMonthlyStats();
      
      return {
        today: {
          calls: todayUsage.psa_api_calls,
          limit: todayUsage.psa_api_limit,
          remaining: todayUsage.getRemainingPsaCalls(),
          percentage: todayUsage.getUsagePercentage('psa'),
          successful: todayUsage.psa_successful_calls,
          failed: todayUsage.psa_failed_calls,
          avgResponseTime: todayUsage.psa_avg_response_time,
          maxResponseTime: todayUsage.psa_max_response_time
        },
        weekly: weeklyStats,
        monthly: monthlyStats
      };
    } catch (error) {
      logError(error, 'Error getting API usage statistics');
      throw error;
    }
  }
}

module.exports = new PsaApiService();