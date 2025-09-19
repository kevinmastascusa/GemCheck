# PSA Pre-Grader Backend API

A comprehensive PSA-style trading card pre-grading backend with PostgreSQL database integration, computer vision analysis, and real-time market data.

## 🚀 Features

### Core Functionality
- **PSA API Integration**: Fetch real PSA certificate data with rate limiting
- **Computer Vision Analysis**: Advanced card condition assessment using GemCheck
- **Market Price Analysis**: Real-time price tracking and comparison
- **Population Reports**: Statistical analysis of card populations
- **Batch Processing**: Handle multiple cards efficiently

### Advanced Features
- **AI-Powered Grading**: Machine learning grade predictions
- **Market Trend Analysis**: Price history and outlier detection
- **Export Capabilities**: PDF reports, CSV data, JSON exports
- **Real-time Updates**: Live grade calculations and market data

## 🏗️ Architecture

```
server/
├── config/             # Database & logging configuration
├── models/             # PostgreSQL/Sequelize models
├── services/           # PSA API integration
├── controllers/        # Route handlers
├── routes/             # API endpoints
└── scripts/            # Database setup utilities
```

## 📋 Prerequisites

- **Node.js** 16+ and npm 8+
- **PostgreSQL** 12+ database
- **PSA API Token** (sign up at PSA)

## 🛠️ Installation

### 1. Install Dependencies
```bash
npm install
```

### 2. Database Setup
```bash
# Create PostgreSQL database
createdb psa_pregrader

# Run database setup
npm run setup-db
```

### 3. Environment Configuration
```bash
# Copy environment template
cp .env.example .env
```

Required environment variables:
```env
# PSA API Configuration
PSA_API_TOKEN=your_psa_api_token_here
PSA_DAILY_LIMIT=100

# Database Configuration
DB_HOST=localhost
DB_NAME=psa_pregrader
DB_USER=your_username
DB_PASSWORD=your_password

# Server Configuration
PORT=3001
NODE_ENV=development
```

## 🚦 Usage

```bash
# Development mode with auto-reload
npm run dev

# Production mode
npm start
```

The API will be available at `http://localhost:3001`

Visit `http://localhost:3001/api` for complete API documentation.

## 📚 API Endpoints

### PSA Card Management
```bash
# Fetch card from PSA API
POST /api/psa/fetch/:certNumber

# Get stored card data
GET /api/psa/card/:certNumber

# Search cards by criteria
GET /api/psa/search?player=Jordan&year=1986

# Batch fetch multiple cards
POST /api/psa/batch-fetch

# Get population analysis
GET /api/psa/population/:certNumber

# Database statistics
GET /api/psa/stats
```

### Pre-Grading Evaluation
```bash
# Evaluate card for grading
POST /api/pregrade/evaluate

# Get evaluation history
GET /api/pregrade/history

# Get accuracy statistics
GET /api/pregrade/accuracy
```

### API Usage Tracking
```bash
# Check current usage status
GET /api/usage/status

# Get detailed usage statistics
GET /api/usage/stats?period=week
```

### Price Analysis
```bash
# Get market analysis
GET /api/prices/market/:cardName

# Get price history
GET /api/prices/history

# Compare prices across grades
GET /api/prices/compare

# Detect price outliers
GET /api/prices/outliers
```

## 📊 Database Schema

### PSA Cards (`psa_cards`)
- cert_number (VARCHAR, UNIQUE)
- grade (INTEGER)
- card_name (VARCHAR)
- player_name (VARCHAR)
- year (INTEGER)
- brand (VARCHAR)
- total_population (INTEGER)
- population_higher (INTEGER)
- raw_api_data (JSONB)

### Pre-grading Evaluations (`pregrading_evaluations`)
- card_name (VARCHAR)
- centering_score (DECIMAL)
- edges_score (DECIMAL)
- corners_score (DECIMAL)
- surface_score (DECIMAL)
- predicted_grade (INTEGER)
- confidence_score (DECIMAL)
- analysis_factors (JSONB)

### API Usage Tracking (`api_usage_tracking`)
- usage_date (DATE, UNIQUE)
- psa_api_calls (INTEGER)
- psa_successful_calls (INTEGER)
- psa_failed_calls (INTEGER)
- hourly_distribution (JSONB)

### Price Comparisons (`price_comparisons`)
- card_name (VARCHAR)
- psa_grade (INTEGER)
- sale_price (DECIMAL)
- sale_date (DATE)
- marketplace (VARCHAR)
- sale_type (ENUM)

## 🔒 Security Features

- **Rate Limiting**: 100 requests per 15-minute window
- **Input Validation**: Joi schema validation
- **SQL Injection Protection**: Sequelize ORM parameterized queries
- **CORS Configuration**: Restricted origin access
- **Helmet Security**: HTTP security headers
- **API Token Protection**: Environment variable storage

## 📈 Monitoring & Logging

### Logging Levels
- **DEBUG**: Detailed operation logs
- **INFO**: General application events
- **WARN**: Potential issues and rate limits
- **ERROR**: Application errors and failures

### Log Files
- `logs/app.log`: General application logs
- `logs/error.log`: Error-only logs

## 📄 API Rate Limits

| Service | Limit | Window | Notes |
|---------|--------|---------|--------|
| PSA API | 100 calls | Daily | Free tier limit |
| General API | 100 requests | 15 minutes | Per IP address |
| Batch Processing | 50 certificates | Per request | Maximum batch size |
| Image Uploads | 5 files | Per request | 10MB total limit |

---

**Backend infrastructure ready for revolutionary 3D frontend!** 🚀