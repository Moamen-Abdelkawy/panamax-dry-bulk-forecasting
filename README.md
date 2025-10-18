# Panamax Dry Bulk Freight Rate Forecasting

A comprehensive hybrid time series-machine learning forecasting system for predicting daily freight rates for Panamax bulk carriers in Atlantic (P1A_82) and Pacific (P3A_82) routes using multi-source shipping intelligence data.

## Overview

This project develops an interpretable hybrid forecasting framework that combines classical econometric time series methods with modern machine learning techniques to predict Panamax vessel time charter equivalent (TCE) rates across multiple horizons (1, 5, 10, and 20 days ahead). The framework integrates GARCH-based volatility modeling and explainable AI techniques to provide both return and risk forecasts with clear interpretability.

**Target Variables:**
- **P1A_82:** Atlantic route Panamax spot rates ($/day)
- **P3A_82:** Pacific route Panamax spot rates ($/day)

## Research Questions

### Primary Research Question
Can a hybrid time series-machine learning model provide a statistically significant improvement in forecasting accuracy for Panamax timecharter rates compared to benchmark models, including traditional econometric approaches and industry-standard forward assessments?

This addresses the core methodological challenge of developing models that capture both linear temporal patterns (through time series methods) and complex non-linear relationships (through machine learning algorithms), with comparison against industry benchmarks, particularly the Baltic Exchange Forward Assessment (BFA).

### Secondary Research Question
To what extent can GARCH models capture the volatility dynamics of Panamax freight rates, and how does the incorporation of volatility forecasting enhance the overall predictive framework?

This investigates whether time-varying volatility characteristics can be effectively modeled using GARCH specifications and whether volatility predictions provide additional value for risk management and commercial decision-making.

### Tertiary Research Question
What are the key drivers of the hybrid model's forecasts, and do they align with established market fundamentals and domain expertise, as revealed through explainable AI techniques?

This addresses model interpretability through SHAP (SHapley Additive exPlanations) and other explainable AI techniques, enabling validation against market knowledge and facilitating trust among industry practitioners.

## Proposed Solution

The research proposes an **interpretable hybrid forecasting framework** that systematically integrates:

1. **ARIMA/SARIMA models** for capturing linear temporal patterns with theoretical foundation
2. **XGBoost/LightGBM algorithms** for identifying complex non-linear relationships and interaction effects
3. **GARCH-based volatility modeling** for time-varying risk characteristics
4. **SHAP-based explainable AI** for model interpretability and validation

This approach leverages the complementary strengths of both econometric and machine learning paradigms while mitigating their individual limitations, ensuring both predictive accuracy and practical interpretability for commercial applications.

## Project Structure

```
panamax-dry-bulk-forecasting/
├── 01_main_pipeline.ipynb              # Data acquisition and consolidation
├── 02_feature_engineering_eda.ipynb    # Feature engineering and exploratory analysis
├── 03_data_preparation.ipynb           # Data alignment and preparation
├── 04_arima_sarima_models.ipynb        # ARIMA/SARIMA models
├── 05_garch_volatility_models.ipynb    # GARCH/volatility modeling
├── 06_xgboost_models.ipynb             # XGBoost implementation
├── 07_xgboost_core_features.ipynb      # Core feature optimization
├── 08_feature_selection_lasso_ridge.ipynb # Feature selection methods
├── 09_model_diagnostics.ipynb          # Model evaluation and diagnostics
├── environment.yml                      # Conda environment specification
├── requirements.txt                     # Python package requirements
└── README.md                            # Project documentation
```

## Data Sources

The project integrates four primary data sources spanning March 2021 to October 2025:

1. **Bunker Prices** - Daily fuel cost data (VLSFO, MGO)
2. **Baltic Exchange** - 45 shipping indices including BPI, BDI, BCI, and FFA trading volumes
3. **BFA (Baltic Forward Assessments)** - Forward contract prices for Atlantic (P1EA) and Pacific (P3EA) routes
4. **Clarksons Shipping Intelligence** - 634 time series across supply metrics, demand indicators, and economic fundamentals

## Features

The forecasting system utilizes 80+ engineered features across multiple categories:

### Market Fundamentals
- Baltic dry bulk indices (BDI, BPI, BCI)
- FFA market activity (volumes, open interest)
- Operating cost metrics (PDOPEX)

### Term Structure Features
- Forward-spot basis spreads
- Term structure slope and curvature
- Contango/backwardation indicators

### Supply Indicators
- Fleet utilization and idle capacity
- Orderbook, deliveries, and demolition
- Regional deployment indices

### Demand Indicators
- Regional trade volumes (grain, iron ore, coal)
- Industrial production growth (USA, Germany, China, India, Japan)
- Port activity and congestion metrics

### Cost Factors
- Bunker fuel prices (VLSFO, MGO)

## Methodology

### Data Pipeline
1. **Data Acquisition:** Multi-source data integration with intelligent frequency detection
2. **Feature Engineering:** Term structure metrics and temporal alignment
3. **Data Preparation:** Proper lagging, temporal splits, and feature scaling

### Modeling Approaches

| Model Type | Purpose | Implementation |
|------------|---------|----------------|
| **ARIMA/ARIMAX** | Capture linear temporal patterns | Exogenous variables with proper differencing |
| **GARCH/EGARCH** | Model volatility dynamics | Conditional heteroskedasticity for risk forecasting |
| **XGBoost** | Identify non-linear relationships | Tree ensemble with SHAP explainability |
| **LightGBM** | Alternative ML approach | Gradient boosting for comparison |
| **Ridge/Lasso** | Feature selection | Regularization-based dimensionality reduction |

### Forecasting Horizons
- **h=1:** 1-day ahead (nowcast)
- **h=5:** 5-day ahead (weekly)
- **h=10:** 10-day ahead (bi-weekly)
- **h=20:** 20-day ahead (monthly)

### Benchmark Comparison
Model performance is evaluated against:
- Traditional econometric models (ARIMA/SARIMA)
- Industry-standard Baltic Forward Assessments (BFA)
- Naive persistence and historical average baselines

## Technical Stack

### Core Dependencies
- **Python:** 3.11+
- **Data Processing:** pandas 2.2.0+, numpy 2.0.0+, scipy 1.11.0+
- **Machine Learning:** scikit-learn 1.4.0+, xgboost 2.0.0+, lightgbm 4.0.0+
- **Time Series:** statsmodels 0.14.0+, arch 6.3.0+ (GARCH)
- **Explainability:** SHAP 0.44.0+
- **Visualization:** matplotlib 3.8.0+, seaborn 0.13.0+

See `requirements.txt` and `environment.yml` for complete dependency specifications.

## Installation

### Using Conda (Recommended)
```bash
conda env create -f environment.yml
conda activate panamax-forecasting
```

### Using pip
```bash
pip install -r requirements.txt
```

## Usage

Execute notebooks in sequential order:

1. **Data Acquisition:** Run `01_main_pipeline.ipynb` to fetch and consolidate data sources
2. **Feature Engineering:** Run `02_feature_engineering_eda.ipynb` for feature creation and exploratory analysis
3. **Data Preparation:** Run `03_data_preparation.ipynb` for temporal alignment and train/test splits
4. **Modeling:** Execute modeling notebooks (`04-08`) for different forecasting approaches
5. **Evaluation:** Run `09_model_diagnostics.ipynb` for comprehensive model validation

## Key Features

### Data Quality
- Rigorous temporal alignment preventing data leakage
- Publication lag handling for weekly/monthly data
- Outlier-aware scaling and missing value treatment
- Business-day calendar alignment

### Advanced Techniques
- Multi-horizon forecasting with separate model optimization
- Term structure engineering from forward curve data
- Feature importance analysis via SHAP
- Stationarity testing and transformation
- Comprehensive model diagnostics

### Model Interpretability
- SHAP-based feature attribution for machine learning models
- Validation against established market fundamentals
- Transparent feature engineering process
- Clear documentation of modeling decisions

## Data Directory Structure

```
data/
├── raw/                                # Original data sources
│   ├── bunker/                        # Fuel price data
│   ├── baltic_exchange/               # Baltic indices and FFAs
│   └── clarksons/                     # Shipping intelligence files
├── processed/                         # Processed datasets
│   ├── pipeline/                      # Canonical data files
│   ├── *_ml_features.csv             # ML-ready feature sets
│   ├── *_arimax_core.csv             # ARIMA/GARCH feature sets
│   └── scalers/                       # Fitted scaler objects
└── splits/                            # Train/validation/test splits
```

## Project Status

This repository represents the **first iteration** of the forecasting system, establishing:
- Complete data acquisition pipeline
- Comprehensive feature engineering framework
- Multiple modeling approaches (econometric and machine learning)
- Rigorous evaluation methodology
- Explainable AI framework

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- **Moamen Abdelkawy**
- **Manuel Quinto Sabelli**

## Academic Context

This project is the Capstone Project for the **Master of Science in Financial Engineering** at **WorldQuant University**.

## Acknowledgments

- Data providers: Baltic Exchange, Clarksons Research, Ship & Bunker
- Industry Support: National Navigation Company
- WorldQuant University MScFE Program
