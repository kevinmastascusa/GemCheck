# PSA Pre-Grader Backend & Pokemon TCG Database

A comprehensive PSA-style trading card pre-grading backend with PostgreSQL database integration, Pokemon TCG data repository, computer vision analysis, and real-time market data.

## 🚀 Features

### Core Functionality
- **Pokemon TCG Database**: Complete Pokemon card data from official repository (~50,000+ cards)
- **Card Image Management**: Automated download and storage of high-quality card images
- **PSA API Integration**: Fetch real PSA certificate data with rate limiting
- **Computer Vision Analysis**: Advanced card condition assessment using real Pokemon cards
- **Holographic Analysis**: Specialized detection and grading for holographic cards
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
PSA PREGRADER/
├── scripts/                    # Database and data management
│   ├── setup_database.py      # PostgreSQL database setup
│   ├── fetch_pokemon_tcg_data.py  # Pokemon TCG data fetcher
│   ├── query_cards.py          # Database query utilities
│   └── test_database.py        # Database testing
├── app/                        # Main application
│   ├── pokemon/                # Pokemon-specific analysis
│   ├── metrics/                # Grading analysis modules
│   └── main.py                 # Streamlit UI
├── tests/                      # Unit tests with real card data
│   ├── test_centering_real_cards.py
│   └── test_surface_real_cards.py
├── data/                       # Card images and datasets
└── server/                     # Backend API (existing)
    ├── config/                 # Database & logging configuration
    ├── models/                 # PostgreSQL/Sequelize models
    ├── services/               # PSA API integration
    ├── controllers/            # Route handlers
    └── routes/                 # API endpoints
```

## 📋 Prerequisites

- **Python** 3.8+ with pip
- **PostgreSQL** 17+ database
- **Node.js** 16+ and npm 8+ (for frontend)
- **PSA API Token** (sign up at PSA)
- **Internet connection** for fetching Pokemon TCG data (~10GB download)

## 🛠️ Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Setup Pokemon TCG Database
```bash
# Interactive database setup
python scripts/setup_database.py

# This creates database and .env file automatically
```

### 3. Test Database Connection
```bash
# Quick test with sample data (2 sets)
python scripts/test_database.py
```

### 4. Fetch Pokemon TCG Data (Optional)
```bash
# Full dataset (~50,000 cards, ~10GB images)
python scripts/fetch_pokemon_tcg_data.py --db-url "$DATABASE_URL"

# Testing with limited data (recommended first)
python scripts/fetch_pokemon_tcg_data.py --db-url "$DATABASE_URL" --limit-sets 5
```

### 5. Install Frontend Dependencies (Optional)
```bash
cd frontend
npm install
```

### Environment Configuration

The setup script creates `.env` automatically, but you can customize:

```env
# Database Configuration (auto-generated)
DATABASE_URL="postgresql://postgres:password@localhost:5432/psa_pregrader"

# Data Storage
CARD_IMAGES_DIR="data/card_images"

# PSA API Configuration (optional)
PSA_API_TOKEN=your_psa_api_token_here
PSA_DAILY_LIMIT=100

# Server Configuration (for backend API)
PORT=3001
NODE_ENV=development
```

## 🚦 Usage

### Python Application
```bash
# Run main PSA grading application
python run.py
# Opens at http://localhost:8501

# Run with specific card analysis
python test_pdf_simple.py  # Tests PDF generation with real Charizard
```

### Database Queries
```bash
# Show database statistics
python scripts/query_cards.py --stats

# Get random cards for testing
python scripts/query_cards.py --random 10

# Get holographic cards
python scripts/query_cards.py --holo 20

# Get cards from specific set
python scripts/query_cards.py --set "Base Set"

# Export test dataset
python scripts/query_cards.py --export test_dataset.json --count 100
```

### Unit Testing with Real Cards
```bash
# Run tests with real Pokemon card data
python -m pytest tests/test_centering_real_cards.py -v
python -m pytest tests/test_surface_real_cards.py -v
```

### Backend API (Optional)
```bash
cd server
npm run dev  # Development mode
npm start    # Production mode
```

The API will be available at `http://localhost:3001`

## 📚 Pokemon TCG Database Schema

### Core Tables

#### pokemon_cards_full
Complete Pokemon card data from official TCG repository:
```sql
id VARCHAR PRIMARY KEY,           -- Unique card identifier (e.g., "base1-4")
name VARCHAR,                     -- Card name (e.g., "Charizard")
set_name VARCHAR,                 -- Set name (e.g., "Base Set")
set_id VARCHAR,                   -- Set identifier (e.g., "base1")
number VARCHAR,                   -- Card number (e.g., "4/102")
rarity VARCHAR,                   -- Rarity (e.g., "Rare Holo")
hp INTEGER,                       -- Hit points
types TEXT[],                     -- Pokemon types (e.g., ["Fire"])
supertype VARCHAR,                -- "Pokémon", "Trainer", "Energy"
subtypes TEXT[],                  -- Card subtypes
artist VARCHAR,                   -- Card artist
images JSONB,                     -- Image URLs (small, large, raw)
abilities JSONB,                  -- Pokemon abilities
attacks JSONB,                    -- Attack information
national_pokedex_number INTEGER,  -- Pokedex number
release_date DATE,                -- Set release date
```

#### card_images
Image storage and management:
```sql
id SERIAL PRIMARY KEY,
card_id VARCHAR REFERENCES pokemon_cards_full(id),
image_type VARCHAR,               -- 'small', 'large', 'raw'
image_url VARCHAR,                -- Original URL
local_path VARCHAR,               -- Downloaded file path
file_size BIGINT,                 -- File size in bytes
width INTEGER,                    -- Image width
height INTEGER,                   -- Image height
checksum VARCHAR,                 -- File integrity check
downloaded_at TIMESTAMP           -- Download timestamp
```

#### pokemon_sets
Set information and metadata:
```sql
id VARCHAR PRIMARY KEY,           -- Set identifier
name VARCHAR,                     -- Set name
series VARCHAR,                   -- Series name
total INTEGER,                    -- Total cards in set
release_date DATE,                -- Release date
legalities JSONB,                 -- Format legalities
images JSONB                      -- Set symbol images
```

## 📚 Python API & Usage Examples

### Database Interface
```python
from scripts.query_cards import CardDatabase

# Connect to database
db = CardDatabase(os.environ['DATABASE_URL'])

# Get random cards
cards = db.get_random_cards(10, rarity="Rare Holo")

# Get cards from specific set
base_set_cards = db.get_cards_by_set("Base Set")

# Get holographic cards for testing
holo_cards = db.get_holo_cards(20)

# Get cards by era
vintage_cards = db.get_cards_by_era(1998, 2003)

# Export test dataset
dataset = db.export_test_dataset("test_cards.json", 100)
```

### Card Analysis with Real Data
```python
import cv2
from app.pokemon import analyze_pokemon_card, generate_pokemon_report

# Get a real card from database
cards = db.get_random_cards(1, rarity="Rare Holo")
card = cards[0]

# Load the card image
if card['image_path']:
    image = cv2.imread(card['image_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run analysis
    result = analyze_pokemon_card(image, include_holo_analysis=True)
    
    # Generate PDF report
    success = generate_pokemon_report(image, result, "grading_report.pdf")
```

## 📚 Backend API Endpoints (Existing)

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

### Pokemon TCG Data (New)
```bash
# Get card data by ID
GET /api/pokemon/card/:cardId

# Search Pokemon cards
GET /api/pokemon/search?name=Charizard&set=Base%20Set

# Get holographic cards
GET /api/pokemon/holo?limit=20

# Get cards by era
GET /api/pokemon/era?start=1998&end=2003

# Get database statistics
GET /api/pokemon/stats

# Export dataset
POST /api/pokemon/export
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

## 📊 Data Statistics & Usage

### Database Size
- **Sets**: ~400 Pokemon sets (1998-2025)
- **Cards**: ~50,000+ unique Pokemon cards  
- **Images**: ~150,000+ images (small/large/raw formats)
- **Storage**: ~10GB total (500MB database + 9.5GB images)

### Data Sources
- **Pokemon TCG API**: https://api.pokemontcg.io/v2
- **Pokemon TCG Data Repository**: https://github.com/PokemonTCG/pokemon-tcg-data
- **License**: MIT License with proper attribution

### Rarity Distribution
```
Common: ~15,000 cards (30%)
Uncommon: ~12,000 cards (24%)  
Rare: ~8,000 cards (16%)
Rare Holo: ~6,000 cards (12%)
Ultra Rare: ~5,000 cards (10%)
Secret Rare: ~4,000 cards (8%)
```

### Era Coverage
```
Vintage (1998-2001): Base Set, Jungle, Fossil, Team Rocket, Neo series
E-Card (2001-2003): E-Card series
EX (2003-2007): EX Ruby & Sapphire through EX Power Keepers
Diamond & Pearl (2007-2011): DP series
Black & White (2011-2014): BW series  
XY (2014-2017): XY series
Sun & Moon (2017-2020): SM series
Sword & Shield (2020-2022): SWSH series
Scarlet & Violet (2023+): SV series
```

## 🧪 Testing with Real Cards

The database enables comprehensive testing with real Pokemon cards:

### Centering Analysis
```python
# Test centering with various card eras
vintage_cards = db.get_cards_by_era(1998, 2003, 50)  # Vintage printing
modern_cards = db.get_cards_by_era(2020, 2025, 50)   # Modern printing

# Compare centering quality across eras
for card in vintage_cards:
    result = analyze_centering(load_image(card['image_path']))
    print(f"Vintage {card['name']}: {result.centering_score:.1f}")
```

### Holographic Analysis
```python
# Get real holographic cards for holo pattern testing
holo_cards = db.get_holo_cards(100)

for card in holo_cards:
    if card['rarity'] in ['Rare Holo', 'Rare Holo EX', 'Rare Holo GX']:
        image = load_image(card['image_path'])
        holo_analysis = analyze_holographic_patterns(image)
        print(f"{card['name']}: {holo_analysis.pattern_type}")
```

### Surface Quality Analysis
```python
# Test surface analysis across different card conditions
for rarity in ['Common', 'Rare Holo', 'Ultra Rare']:
    cards = db.get_random_cards(20, rarity=rarity)
    
    for card in cards:
        image = load_image(card['image_path'])
        surface_result = analyze_surface(image)
        print(f"{rarity} {card['name']}: {surface_result.surface_score:.1f}")
```

## 📊 Legacy Database Schema (Existing)

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

## 🔄 Data Management

### Fetching Data
```bash
# Full dataset (production)
python scripts/fetch_pokemon_tcg_data.py --db-url "$DATABASE_URL"

# Limited for testing
python scripts/fetch_pokemon_tcg_data.py --db-url "$DATABASE_URL" --limit-sets 10

# Verbose logging for debugging  
python scripts/fetch_pokemon_tcg_data.py --db-url "$DATABASE_URL" --verbose
```

### Querying Data
```bash
# Database statistics
python scripts/query_cards.py --stats

# Random sampling
python scripts/query_cards.py --random 50 --rarity "Rare Holo"

# Set-specific queries
python scripts/query_cards.py --set "Base Set"
python scripts/query_cards.py --set "Hidden Fates"

# Era-based queries
python scripts/query_cards.py --era 1998 2003  # Vintage
python scripts/query_cards.py --era 2020 2025  # Modern

# Export datasets
python scripts/query_cards.py --export vintage_dataset.json --era 1998 2003 --count 500
python scripts/query_cards.py --export holo_dataset.json --holo 200
```

### Data Integrity
```bash
# Test database setup
python scripts/test_database.py

# Verify image downloads
python scripts/query_cards.py --stats | grep "downloaded_images"

# Check for missing images
python -c "
from scripts.query_cards import CardDatabase
db = CardDatabase('$DATABASE_URL')
conn = db.connect()
with conn.cursor() as cur:
    cur.execute('SELECT COUNT(*) FROM card_images WHERE local_path IS NULL')
    print(f'Missing images: {cur.fetchone()[0]}')
"
```

## 🎯 Integration Examples

### Real Card Testing Pipeline
```python
def test_grading_accuracy():
    \"\"\"Test grading accuracy using real cards with known conditions.\"\"\"
    db = CardDatabase(os.environ['DATABASE_URL'])
    
    # Get diverse sample
    test_cards = []
    test_cards.extend(db.get_holo_cards(50))        # Holo challenges
    test_cards.extend(db.get_cards_by_era(1998, 2003, 50))  # Vintage printing
    test_cards.extend(db.get_cards_by_era(2020, 2025, 50))  # Modern quality
    
    results = []
    for card in test_cards:
        if card['image_path']:
            image = cv2.imread(card['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run full analysis
            analysis = analyze_pokemon_card(image, include_holo_analysis=True)
            
            results.append({
                'card_id': card['id'],
                'name': card['name'],
                'set': card['set_name'], 
                'rarity': card['rarity'],
                'era': get_era_from_date(card['release_date']),
                'predicted_grade': analysis.overall_grade,
                'centering': analysis.centering_grade,
                'surface': analysis.surface_grade,
                'edges': analysis.edges_grade,
                'corners': analysis.corners_grade
            })
    
    return results
```

### Card Recognition Training
```python
def create_training_dataset():
    \"\"\"Create training dataset for card recognition.\"\"\"
    db = CardDatabase(os.environ['DATABASE_URL'])
    
    # Get representative samples from each era
    eras = [
        (1998, 2003, 'vintage'),
        (2003, 2007, 'ex'),  
        (2020, 2025, 'modern')
    ]
    
    training_data = []
    for start, end, era_name in eras:
        cards = db.get_cards_by_era(start, end, 1000)
        
        for card in cards:
            if card['image_path']:
                training_data.append({
                    'image_path': card['image_path'],
                    'labels': {
                        'name': card['name'],
                        'set': card['set_name'],
                        'number': card['number'],
                        'rarity': card['rarity'],
                        'era': era_name,
                        'types': card['types'],
                        'hp': card['hp']
                    }
                })
    
    return training_data
```

---

**Complete Pokemon TCG database with 50,000+ real cards ready for AI-powered grading!** 🚀