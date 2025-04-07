"""
NVIDIA Stock Forecasting Model with AGI Timeline Analysis.

This module implements the main forecasting model that combines
traditional stock analysis with AGI development probabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging

from ..data.stock_data import fetch_stock_data, calculate_technical_indicators
from ..models.agi_probability import AGIProbabilityModel
from ..anthropic.claude_client import AnthropicClient
from ..utils.config import load_config

# Set up logging
logger = logging.getLogger(__name__)

class NvidiaStockForecast:
    """
    A forecasting model for NVIDIA stock that incorporates AGI timeline
    probabilities and AI landscape analysis.
    """

    def __init__(self, config_path=None):
        """
        Initialize the NVIDIA stock forecasting model.
        
        Args:
            config_path (str, optional): Path to configuration file.
                If None, default configuration will be used.
        """
        # Load configuration
        self.config = load_config(config_path)
        self.stock_data = None
        self.model = None
        self.ai_sentiment_score = None
        self.agi_probability_factor = None
        self.forecast_results = None
        
        # Initialize Anthropic client if API key provided
        self.anthropic_client = None
        if self.config.get('anthropic_api_key'):
            self.anthropic_client = AnthropicClient(self.config['anthropic_api_key'])
            logger.info("Anthropic client initialized")
        else:
            logger.warning("No Anthropic API key provided, running without Claude analysis")
        
        # Initialize the AGI probability model
        self.agi_model = AGIProbabilityModel(
            self.anthropic_client,
            current_year=self.config.get('current_year', datetime.now().year)
        )
        logger.info("Initialized NVIDIA Stock Forecast model")
    
    def fetch_stock_data(self, years=5, use_cache=True):
        """
        Fetch NVIDIA historical stock data and calculate technical indicators.
        
        Args:
            years (int): Number of years of historical data to fetch
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            pd.DataFrame: The processed stock data
        """
        logger.info(f"Fetching NVIDIA stock data for the past {years} years")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*years)
        
        # Use the data module to fetch stock data
        self.stock_data = fetch_stock_data(
            ticker="NVDA",
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            cache_dir=self.config.get('data_cache_dir', 'data/raw')
        )
        
        # Calculate technical indicators
        self.stock_data = calculate_technical_indicators(self.stock_data)
        logger.info(f"Retrieved {len(self.stock_data)} days of trading data")
        
        return self.stock_data
    
    def analyze_ai_landscape(self):
        """
        Analyze AI landscape using Anthropic Claude and the AGI probability model.
        
        Returns:
            dict: Analysis results including AGI probabilities and AI landscape info
        """
        logger.info("Analyzing AI landscape and AGI progress")
        
        # Default values if Claude is not available
        nvidia_position = "NVIDIA dominates the AI chip market with its GPUs."
        ai_advancements = "Recent advances include larger multimodal models."
        agi_timeline = "Estimates range from 2030 to 2050 depending on definitions."
        
        # Use Claude to extract information if available
        if self.anthropic_client:
            try:
                nvidia_position = self.agi_model.get_claude_analysis("nvidia_position")
                ai_advancements = self.agi_model.get_claude_analysis("ai_advancements")
                agi_timeline = self.agi_model.get_claude_analysis("agi_timeline")
                logger.info("Successfully retrieved AI landscape analysis from Claude")
            except Exception as e:
                logger.error(f"Error retrieving analysis from Claude: {e}")
                logger.info("Using default values instead")
        
        # Run the AGI probability model
        logger.info("Running AGI probability model")
        self.agi_model.run_simplified_model()
        
        # Get the AGI probability factor for current year + 5
        years = np.arange(self.agi_model.current_year, self.agi_model.current_year + 15)
        agi_probs = self.agi_model.get_agi_probability_by_year(years)
        
        # Get the probability for 5 years from now
        five_year_prob = agi_probs.loc[agi_probs['year'] == self.agi_model.current_year + 5, 'cumulative_probability'].iloc[0]
        self.agi_probability_factor = five_year_prob
        
        # Calculate AI sentiment score based on position in AI market
        # In a real implementation, this would use NLP on news articles, etc.
        self.ai_sentiment_score = 0.75  # Placeholder
        
        logger.info(f"AI Sentiment Score: {self.ai_sentiment_score:.2f}")
        logger.info(f"AGI Probability Factor (5-year): {self.agi_probability_factor:.2f}")
        
        return {
            'ai_sentiment': self.ai_sentiment_score,
            'agi_probability': self.agi_probability_factor,
            'nvidia_position': nvidia_position,
            'ai_advancements': ai_advancements,
            'agi_timeline': agi_timeline
        }
    
    def prepare_features(self):
        """
        Prepare features for the prediction model.
        
        Returns:
            pd.DataFrame: Features ready for model training
        """
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data() first.")
            
        logger.info("Preparing features for model training")
        
        # Select base features
        feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                        'MA50', 'MA200', 'RSI', 'Volatility']
        features = self.stock_data[feature_cols].copy()
        
        # Add lag features
        for lag in [1, 5, 10, 21]:  # 1 day, 1 week, 2 weeks, 1 month
            features[f'Close_lag_{lag}'] = features['Close'].shift(lag)
            features[f'Volume_lag_{lag}'] = features['Volume'].shift(lag)
        
        # Add price momentum features
        for period in [5, 21, 63]:  # 1 week, 1 month, 3 months
            features[f'momentum_{period}'] = features['Close'].pct_change(period)
        
        # Incorporate AI landscape features if available
        if self.ai_sentiment_score is not None and self.agi_probability_factor is not None:
            # Extend these constant values to all rows
            features['ai_sentiment'] = self.ai_sentiment_score
            features['agi_probability'] = self.agi_probability_factor
        
        # Create target variable (next 30 day returns)
        features['target_30d_return'] = features['Close'].shift(-30) / features['Close'] - 1
        
        # Drop NaN values
        features = features.dropna()
        logger.info(f"Prepared {len(features)} rows of feature data")
        
        return features
    
    def train_model(self, test_size=0.2, random_state=42):
        """
        Train the stock prediction model.
        
        Args:
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
        
        Returns:
            tuple: (model, train_score, test_score)
        """
        logger.info("Training the stock prediction model")
        
        # Prepare features
        features_df = self.prepare_features()
        
        # Split data
        X = features_df.drop('target_30d_return', axis=1)
        y = features_df['target_30d_return']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False, random_state=random_state
        )
        
        # Get model parameters from config
        model_params = self.config.get('model_params', {})
        n_estimators = model_params.get('n_estimators', 100)
        max_depth = model_params.get('max_depth', None)
        
        # Train Random Forest model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        logger.info(f"Model R² on training data: {train_score:.4f}")
        logger.info(f"Model R² on testing data: {test_score:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        logger.info("\nTop 10 important features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"{row['Feature']}: {row['Importance']:.4f}")
        
        self.model = model
        return model, train_score, test_score
    
    def forecast_future(self, days=365):
        """
        Forecast future stock prices.
        
        Args:
            days (int): Number of days to forecast
            
        Returns:
            pd.DataFrame: DataFrame with forecasted prices
        """
        logger.info(f"Forecasting NVIDIA stock for the next {days} days")
        
        if self.model is None or self.stock_data is None:
            raise ValueError("Model not trained or stock data not loaded")
        
        # Get the latest data point
        latest_data = self.stock_data.iloc[-1].copy()
        latest_close = latest_data['Close']
        
        # Get AGI impact factors for future years
        current_year = datetime.now().year
        forecast_years = np.arange(current_year, current_year + (days // 365) + 2)
        impact_data = self.agi_model.get_stock_impact_factor(forecast_years)
        
        # Initialize forecast array
        forecast_dates = pd.date_range(start=self.stock_data.index[-1] + pd.Timedelta(days=1), periods=days)
        forecasted_prices = np.zeros(days)
        forecasted_prices[0] = latest_close
        
        # Create a copy of features for prediction
        feature_names = [col for col in self.prepare_features().columns if col != 'target_30d_return']
        
        # Forecast each step
        for i in range(1, days):
            # Every 30 days, predict the next 30-day return
            if i % 30 == 0:
                # Create a simplified feature vector for prediction
                prediction_features = pd.DataFrame([latest_data], columns=feature_names)
                
                # Add AI landscape features
                prediction_features['ai_sentiment'] = self.ai_sentiment_score
                
                # Adjust AGI probability based on forecast date
                forecast_date = forecast_dates[i]
                forecast_year = forecast_date.year
                
                if forecast_year in impact_data['year'].values:
                    # Get the impact factor for this year
                    impact_factor = impact_data.loc[impact_data['year'] == forecast_year, 'stock_impact_factor'].iloc[0]
                    prediction_features['agi_probability'] = impact_data.loc[impact_data['year'] == forecast_year, 'agi_probability'].iloc[0]
                else:
                    # Use the last available year if we don't have data for this year
                    impact_factor = impact_data['stock_impact_factor'].iloc[-1]
                    prediction_features['agi_probability'] = impact_data['agi_probability'].iloc[-1]
                
                # Predict the next 30-day return
                predicted_return = self.model.predict(prediction_features)[0]
                
                # Adjust return based on AGI impact
                adjusted_return = predicted_return * (1 + impact_factor)
                
                # Update the price with the predicted return (daily compound)
                for j in range(i, min(i+30, days)):
                    daily_factor = (1 + adjusted_return) ** (1/30)
                    forecasted_prices[j] = forecasted_prices[j-1] * daily_factor
            
            # If we've already set this price from a previous 30-day prediction, skip
            elif forecasted_prices[i] == 0:
                # Simple continuation
                forecasted_prices[i] = forecasted_prices[i-1]
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Forecasted_Close': forecasted_prices
        }).set_index('Date')
        
        self.forecast_results = forecast_df
        logger.info(f"Forecast complete. Final price: ${forecast_df['Forecasted_Close'].iloc[-1]:.2f}")
        
        return forecast_df
    
    def plot_forecast(self, forecast_df=None, save_path=None):
        """
        Plot historical data and forecast.
        
        Args:
            forecast_df (pd.DataFrame, optional): Forecast data. If None, uses
                the latest generated forecast.
            save_path (str, optional): Path to save the plot image.
                
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        if forecast_df is None:
            if self.forecast_results is None:
                raise ValueError("No forecast available. Run forecast_future() first.")
            forecast_df = self.forecast_results
            
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data() first.")
        
        logger.info("Plotting stock forecast")
        
        plt.figure(figsize=(15, 8))
        
        # Plot historical data
        plt.plot(self.stock_data.index, self.stock_data['Close'], label='Historical Close Price')
        
        # Plot forecast
        plt.plot(forecast_df.index, forecast_df['Forecasted_Close'], label='Forecasted Close Price', linestyle='--')
        
        # Plot confidence intervals (simplified)
        upper_bound = forecast_df['Forecasted_Close'] * (1 + 0.2 * np.sqrt(np.arange(len(forecast_df)) / 252))
        lower_bound = forecast_df['Forecasted_Close'] * (1 - 0.2 * np.sqrt(np.arange(len(forecast_df)) / 252))
        plt.fill_between(forecast_df.index, lower_bound, upper_bound, alpha=0.2, color='blue')
        
        # Add labels and title
        plt.title('NVIDIA Stock Price Forecast Considering AI Landscape and AGI Developments', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Stock Price ($)', fontsize=14)
        plt.legend()
        plt.grid(True)
        
        # Annotate important milestones on the forecast
        milestone_days = [90, 180, 365]  # 3 months, 6 months, 1 year
        for day in milestone_days:
            if day < len(forecast_df):
                milestone_date = forecast_df.index[day]
                milestone_price = forecast_df.iloc[day]['Forecasted_Close']
                plt.annotate(f"{milestone_date.strftime('%b %Y')}\n${milestone_price:.2f}",
                            xy=(milestone_date, milestone_price),
                            xytext=(10, 30),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved forecast plot to {save_path}")
        
        return plt.gcf()
    
    def plot_agi_probability(self, save_path=None):
        """
        Plot AGI probability and potential impact on stock.
        
        Args:
            save_path (str, optional): Path to save the plot image.
                
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        logger.info("Plotting AGI probability analysis")
        
        years = np.arange(self.agi_model.current_year, self.agi_model.current_year + 20)
        probs = self.agi_model.get_agi_probability_by_year(years)
        impact = self.agi_model.get_stock_impact_factor(years)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel('Year', fontsize=14)
        ax1.set_ylabel('AGI Probability', color=color, fontsize=14)
        ax1.plot(years, probs['cumulative_probability'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Stock Impact Factor', color=color, fontsize=14)
        ax2.plot(years, impact['stock_impact_factor'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('AGI Probability and Potential NVIDIA Stock Impact', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved AGI probability plot to {save_path}")
            
        return fig
    
    def get_forecast_summary(self, forecast_df=None):
        """
        Generate a summary of the forecast results.
        
        Args:
            forecast_df (pd.DataFrame, optional): Forecast data. If None, uses
                the latest generated forecast.
                
        Returns:
            dict: Summary metrics
        """
        if forecast_df is None:
            if self.forecast_results is None:
                raise ValueError("No forecast available. Run forecast_future() first.")
            forecast_df = self.forecast_results
            
        if self.stock_data is None:
            raise ValueError("Stock data not loaded. Call fetch_stock_data() first.")
        
        current_price = self.stock_data['Close'].iloc[-1]
        
        # Calculate metrics for different timeframes
        timeframes = {
            '1_month': 30,
            '3_months': 90, 
            '6_months': 180,
            '1_year': 365
        }
        
        summary = {
            'current_price': current_price,
            'forecast_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        # Add forecasted prices and returns for each timeframe
        for label, days in timeframes.items():
            if days < len(forecast_df):
                price = forecast_df['Forecasted_Close'].iloc[days - 1]
                return_pct = (price / current_price - 1) * 100
                summary[f'{label}_price'] = price
                summary[f'{label}_return_pct'] = return_pct
        
        # Add AGI probabilities for key years
        for year in range(self.agi_model.current_year, self.agi_model.current_year + 30, 5):
            prob = self.agi_model._calculate_cumulative_probability(year)
            summary[f'agi_prob_{year}'] = prob
            
        return summary


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Simple example usage
    forecaster = NvidiaStockForecast()
    forecaster.fetch_stock_data(years=5)
    forecaster.analyze_ai_landscape()
    forecaster.train_model()
    forecast = forecaster.forecast_future(days=365)
    forecaster.plot_forecast(forecast)
    plt.show()