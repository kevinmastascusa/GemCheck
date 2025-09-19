# 🏆 GemCheck - Revolutionary PSA Card Pre-Grading System

<div align="center">
  <img src="resources/GemCheck Logo.png" alt="GemCheck Logo" width="200"/>
  
  **The Future of Card Authentication**
  
  *Professional PSA-style pre-grading with real-time computational photography overlays*
  
  [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
  [![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red)](https://opencv.org/)
</div>

## 🎯 Features

### Core Functionality
- **PSA-Style Grading**: Complete 1-10 grading scale with authentic PSA criteria
- **Computer Vision Analysis**: Advanced OpenCV-based defect detection
- **Transparent Scoring**: Detailed sub-scores with mathematical formulas
- **Visual Evidence**: Overlay images showing detected issues
- **Offline Operation**: No external API calls required

### Analysis Categories
- **Centering**: Edge-based and color-based frame detection
- **Edges**: Whitening detection and nick counting
- **Corners**: Sharpness analysis and wear assessment
- **Surface**: Scratch detection and print line identification  
- **Glare**: Specular highlight detection and penalty calculation

### User Interface
- **Multi-Tab Interface**: Upload, batch processing, configuration, calibration
- **Real-Time Analysis**: Instant results with visual feedback
- **Batch Processing**: Analyze multiple cards simultaneously
- **Export Options**: PDF reports, CSV data, JSON results
- **Configurable Rubric**: Adjust weights and thresholds via YAML files

### Advanced Features
- **Auto-Calibration**: Threshold optimization using reference cards
- **ML Integration**: Plugin architecture for optional PyTorch/ONNX models
- **Synthetic Card Generation**: Create test images with known defects
- **Comprehensive Testing**: Unit tests with synthetic card validation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/psa-card-pregrader.git
cd psa-card-pregrader

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
python run.py
```

The application will automatically start and display the URL. Open your browser to `http://localhost:8501` and start grading cards!

### Generate Test Cards

```bash
python examples/make_synthetic.py --grades 10 9 8 7 --count-per-grade 3
```

## 📁 Project Structure

```
psa-card-pregrader/
├── app/                          # Main application
│   ├── main.py                   # Streamlit UI
│   ├── schema.py                 # Pydantic data models
│   ├── metrics/                  # Analysis modules
│   │   ├── centering.py          # Centering analysis
│   │   ├── edges_corners.py      # Edge/corner analysis
│   │   ├── surface.py            # Surface defect detection
│   │   └── glare.py              # Glare detection
│   ├── ml/                       # Optional ML models
│   │   └── optional_defect_model.py
│   ├── scoring.py                # Grade calculation
│   ├── visualize.py              # Overlay generation
│   ├── report.py                 # Export functionality
│   ├── calibration.py            # Auto-tuning system
│   └── detect_card.py            # Card detection/rectification
├── config/                       # Configuration files
│   ├── weights.yaml              # Scoring weights
│   ├── thresholds.yaml           # Detection parameters
│   └── grade_map.yaml            # Grade definitions
├── tests/                        # Unit tests
├── examples/                     # Example scripts
│   └── make_synthetic.py         # Synthetic card generator
└── requirements.txt              # Dependencies
```

## 🔧 Configuration

### Scoring Weights (`config/weights.yaml`)
```yaml
centering: 0.35    # 35% weight
edges: 0.20        # 20% weight  
corners: 0.20      # 20% weight
surface: 0.25      # 25% weight
```

### Detection Thresholds (`config/thresholds.yaml`)
```yaml
centering:
  max_error_threshold: 0.25      # Maximum centering error
edges:
  whitening_threshold: 0.15      # Edge whitening sensitivity
  acceptable_whitening_percent: 15.0
corners:
  sharpness_threshold: 0.3       # Corner sharpness minimum
surface:
  scratch_threshold: 0.02        # Scratch detection sensitivity
glare:
  highlight_threshold: 0.8       # Glare detection threshold
```

## 🎯 Grade Scale

| Grade | Label | Score Range | Description |
|-------|-------|-------------|-------------|
| 10 | Gem Mint | 97.0+ | Perfect condition |
| 9 | Mint | 92.0+ | Near-perfect with minor imperfections |
| 8 | NM-Mint | 85.0+ | Very high quality, slight wear |
| 7 | Near Mint | 78.0+ | High quality, minor wear |
| 6 | Excellent | 72.0+ | Good condition, moderate wear |
| 5 | VG-EX | 66.0+ | Moderate wear, still collectible |
| 4 | Good | 60.0+ | Significant wear, no major damage |
| 3 | Fair | 54.0+ | Heavy wear, minor damage |
| 2 | Poor | 48.0+ | Extensive wear and damage |
| 1 | Authentic | 0.0+ | Authentic but poor condition |

## 📊 Analysis Methods

### Centering Analysis
- **Edge Detection**: Canny edge detection with Hough transforms
- **Color Segmentation**: HSV-based frame detection
- **Error Calculation**: Mathematical centering error with tolerance bands
- **Formula**: `Score = max(0, 100 × (1 - error / threshold))`

### Surface Defect Detection
- **Morphological Operations**: Opening/closing for defect isolation
- **Connected Components**: Defect region identification
- **Machine Learning**: Optional CNN-based defect classification
- **Scratch Detection**: Line detection with length/orientation filters

### Corner Analysis
- **Sharpness Measurement**: Gradient-based corner quality
- **Wear Assessment**: Color analysis for whitening/damage
- **Individual Scoring**: Per-corner analysis with aggregate scoring

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific module
python -m pytest tests/test_centering.py -v

# Generate test coverage
python -m pytest tests/ --cov=app --cov-report=html
```

## 📈 Calibration

Auto-tune detection parameters using reference cards:

```bash
python -c "
from app.calibration import run_calibration_workflow
result = run_calibration_workflow()
print(f'Improvement: {result.improvement_score:.2%}')
print(f'Accuracy: {result.target_accuracy:.2%}')
"
```

## 🤖 ML Integration

The application supports optional machine learning models:

```python
# Enable ML in config/thresholds.yaml
ml:
  enabled: true
  model_type: "pytorch"  # or "onnx"
  model_path: "models/defect_detector.pth"
  confidence_threshold: 0.5
```

## 📄 Export Formats

### PDF Reports
- Comprehensive analysis with visual overlays
- Grade breakdown and scoring formulas
- Professional formatting for documentation

### CSV Export  
- Batch analysis results in tabular format
- Sub-scores and metadata for statistical analysis

### JSON Export
- Complete analysis data in structured format
- API-friendly for integration with other systems

## 🐛 Troubleshooting

### Common Issues

**OpenCV Installation**
```bash
pip install opencv-python-headless  # For servers without display
```

**Memory Issues with Large Images**
- Images are automatically resized for processing
- Adjust `max_size` in `config/thresholds.yaml`

**Low Detection Accuracy**
- Use calibration system with reference cards
- Adjust thresholds in configuration files
- Ensure good lighting and image quality

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PSA (Professional Sports Authenticator) for grading standards reference
- OpenCV community for computer vision tools
- Streamlit team for the excellent web framework
- PyTorch team for machine learning capabilities
- **[PokemonTCG/pokemon-tcg-data](https://github.com/PokemonTCG/pokemon-tcg-data)** - Official Pokémon TCG data repository providing comprehensive card information, sets, rarities, and mechanics data under MIT License

## ⚠️ Disclaimer

This application provides pre-grading estimates for educational and reference purposes only. Results are not official PSA grades and should not be used for commercial valuation. Always consult professional grading services for official authentication and grading.

---

**Made with ❤️ for the trading card community**