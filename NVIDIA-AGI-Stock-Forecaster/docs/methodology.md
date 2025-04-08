# NVIDIA AGI Stock Forecaster: Methodology

This document describes the methodology used in our NVIDIA stock forecasting model that incorporates AGI timeline predictions.

## Overview

Traditional stock forecasting models typically rely on technical indicators, financial fundamentals, and market sentiment. While these are valuable, they fail to capture the unique relationship between NVIDIA's business prospects and the trajectory of artificial intelligence development.

Our approach integrates:

1. Traditional stock analysis techniques
2. AGI timeline probability modeling
3. AI landscape assessment

## Data Sources

The model uses the following data sources:

- **Stock Data**: Historical NVIDIA (NVDA) pricing data from Yahoo Finance
- **Expert Surveys**: AGI timeline predictions from AI researchers and organizations
- **Compute Trends**: Historical data on AI compute scaling
- **AI Research Developments**: Analysis of recent AI capability breakthroughs
- **Market Sentiment**: Analysis using Anthropic's Claude API

## AGI Probability Modeling

The core innovation of our approach is the principled modeling of AGI timeline probabilities.

### Expert Survey Analysis

We aggregate expert predictions from multiple sources:

- AI Impacts surveys of ML researchers
- Grace et al. survey data
- Metaculus prediction community
- AI industry leader statements

For each survey, we extract the median prediction and confidence intervals, then create a mixture distribution representing the combined expert view.

### Compute Trajectory Analysis

We model AGI timelines based on compute scaling trends:

1. Start with current training compute (estimated at 10^26 FLOP)
2. Estimate required compute for AGI capabilities (10^30 FLOP)
3. Apply annual compute growth rate (estimated at 1.3x annually)
4. Model time-to-AGI as a log-normal distribution to account for uncertainty

This approach is based on AI scaling laws research that shows performance scaling with compute follows predictable patterns.

### Combined Probability Model

Our final AGI timeline distribution combines:

- The expert survey distribution (50% weight)
- The compute trajectory distribution (50% weight)

The result is a probability distribution over years that represents our best estimate of when AGI might arrive. From this, we calculate:

- Expected AGI arrival year
- Confidence intervals
- Cumulative probability by year
- Year-by-year probability

## Stock Forecast Model

### Feature Engineering

The model uses several categories of features:

1. **Technical Indicators**:
   - Moving averages (50-day, 200-day)
   - Relative Strength Index (RSI)
   - Price volatility
   - Trading volume

2. **Lagged Features**:
   - Previous day, week, month close prices
   - Price momentum over multiple timeframes

3. **AI-Specific Features**:
   - AGI probability factors
   - AI sentiment score
   - Market positioning in AI

### Machine Learning Model

We use a Random Forest Regressor for our core predictive model:

- Target variable: 30-day future returns
- Training data: 80% of historical data
- Testing data: 20% of historical data (most recent)
- Hyperparameters: Optimized via config file

### AGI Impact Adjustment

A key innovation is our methodology for mapping AGI probabilities to stock price impacts:

1. **Stock Impact Factor**: We calculate how the probability of AGI affects NVIDIA's stock price using:
   - Cumulative AGI probability
   - Year-specific probability
   - Time-weighted impact factors
   - Economic impact threshold

2. **Future Price Adjustment**: The final forecast incorporates:
   - Base prediction from the Random Forest model
   - AGI impact factor adjustment
   - Uncertainty bands that widen with time

## Forecasting Process

The complete forecasting process follows these steps:

1. Fetch historical NVIDIA stock data
2. Run the AGI probability model to get timeline estimates
3. Analyze the AI landscape (optional, using Claude API)
4. Prepare features, including AGI probability factors
5. Train the RandomForest prediction model
6. Generate day-by-day price forecasts with AGI-adjusted returns
7. Calculate confidence intervals and scenario analyses

## Limitations and Caveats

This model has several limitations to be aware of:

1. **AGI Definition Uncertainty**: There's no universal agreement on what constitutes AGI
2. **Expert Disagreement**: Significant variance in expert opinions on AGI timelines
3. **Black Swan Events**: The model can't predict unexpected breakthroughs or setbacks
4. **Market Irrationality**: Stock prices often move based on sentiment rather than fundamentals
5. **Simplifying Assumptions**: The model assumes a direct relationship between AGI progress and NVIDIA's business success

## Future Work

Future improvements to the model could include:

1. Integration of more robust Bayesian modeling with PyMC
2. More sophisticated Monte Carlo simulations for risk assessment
3. Additional ML model types (LSTM, transformer-based models)
4. More nuanced company-specific factors in the AGI impact model
5. Multi-company analysis to compare NVIDIA with competitors

## References

- Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models"
- Sevilla, J., et al. (2022). "Compute Trends Across Three Eras of Machine Learning"
- Grace, K., et al. (2018). "When Will AI Exceed Human Performance? Evidence from AI Experts"
- AI Impacts. "2022 Expert Survey on Progress in AI"
- Metaculus. "Date of Artificial General Intelligence"