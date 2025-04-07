#!/usr/bin/env python
"""
Generate NVIDIA stock forecast with AGI probability modeling.

This script runs the full NVIDIA forecasting pipeline and generates
visualizations and a summary report.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.stock_forecast import NvidiaStockForecast
from src.utils.config import load_config


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate NVIDIA stock forecast')
    
    parser.add_argument('--config', type=str, default='config/model_params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--years', type=int, default=5,
                        help='Years of historical data to use')
    parser.add_argument('--days', type=int, default=365,
                        help='Days to forecast into the future')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files')
    parser.add_argument('--api-key', type=str,
                        help='Anthropic API key (overrides config file)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--skip-claude', action='store_true',
                        help='Skip Claude analysis (much faster, but less insightful)')
    
    return parser.parse_args()


def setup_logging(log_level, output_dir):
    """Set up logging to both console and file."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(output_dir, f'forecast_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def main():
    """Run the NVIDIA stock forecast pipeline."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up logging
    log_file = setup_logging(args.log_level, args.output_dir)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting NVIDIA forecast generation (Output directory: {args.output_dir})")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override API key if provided
    if args.api_key:
        config['anthropic_api_key'] = args.api_key
    elif args.skip_claude:
        config['anthropic_api_key'] = None
    
    # Create timestamped directory for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")
    
    try:
        # Initialize forecaster
        forecaster = NvidiaStockForecast(config)
        
        # Fetch data
        logger.info(f"Fetching {args.years} years of historical data")
        stock_data = forecaster.fetch_stock_data(years=args.years)
        
        # Analyze AI landscape
        logger.info("Analyzing AI landscape and AGI probabilities")
        ai_analysis = forecaster.analyze_ai_landscape()
        
        # Save AI analysis to file
        ai_analysis_file = os.path.join(run_dir, 'ai_analysis.json')
        with open(ai_analysis_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_analysis = {k: str(v) for k, v in ai_analysis.items()}
            json.dump(serializable_analysis, f, indent=2)
        logger.info(f"Saved AI analysis to {ai_analysis_file}")
        
        # Train model
        logger.info("Training forecasting model")
        model, train_score, test_score = forecaster.train_model()
        
        # Generate forecast
        logger.info(f"Generating {args.days}-day forecast")
        forecast = forecaster.forecast_future(days=args.days)
        
        # Save forecast data
        forecast_file = os.path.join(run_dir, 'forecast_data.csv')
        forecast.to_csv(forecast_file)
        logger.info(f"Saved forecast data to {forecast_file}")
        
        # Generate and save plots
        logger.info("Generating visualizations")
        
        # Stock forecast plot
        forecast_plot = forecaster.plot_forecast(forecast)
        forecast_plot_file = os.path.join(run_dir, 'forecast_plot.png')
        forecast_plot.savefig(forecast_plot_file, dpi=300)
        logger.info(f"Saved forecast plot to {forecast_plot_file}")
        
        # AGI probability plot
        agi_plot = forecaster.plot_agi_probability()
        agi_plot_file = os.path.join(run_dir, 'agi_probability_plot.png')
        agi_plot.savefig(agi_plot_file, dpi=300)
        logger.info(f"Saved AGI probability plot to {agi_plot_file}")
        
        # Generate distributions plot from AGI model
        dist_plot = forecaster.agi_model.plot_distributions()
        dist_plot_file = os.path.join(run_dir, 'agi_distributions_plot.png')
        dist_plot.savefig(dist_plot_file, dpi=300)
        logger.info(f"Saved AGI distributions plot to {dist_plot_file}")
        
        # Generate cumulative probability plot
        cum_plot = forecaster.agi_model.plot_cumulative_probability()
        cum_plot_file = os.path.join(run_dir, 'agi_cumulative_plot.png')
        cum_plot.savefig(cum_plot_file, dpi=300)
        logger.info(f"Saved AGI cumulative probability plot to {cum_plot_file}")
        
        # Generate and save forecast summary
        summary = forecaster.get_forecast_summary(forecast)
        
        # Add additional metadata
        summary['generation_time'] = datetime.now().isoformat()
        summary['historical_years'] = args.years
        summary['forecast_days'] = args.days
        summary['train_score'] = train_score
        summary['test_score'] = test_score
        
        # Save summary to json file
        summary_file = os.path.join(run_dir, 'forecast_summary.json')
        with open(summary_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_summary = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v 
                                  for k, v in summary.items()}
            json.dump(serializable_summary, f, indent=2)
        logger.info(f"Saved forecast summary to {summary_file}")
        
        # Generate a human-readable report
        generate_report(summary, ai_analysis, run_dir)
        
        logger.info(f"Forecast generation complete. All outputs saved to {run_dir}")
        
        # Print key metrics to console
        print("\n=== NVIDIA Forecast Summary ===")
        print(f"Current Price: ${summary['current_price']:.2f}")
        print(f"1-Year Forecast: ${summary['1_year_price']:.2f} ({summary['1_year_return_pct']:.2f}%)")
        print(f"AGI Probability by 2030: {summary['agi_prob_2030']*100:.1f}%")
        print(f"AGI Probability by 2040: {summary['agi_prob_2040']*100:.1f}%")
        print(f"\nFull results saved to: {run_dir}")
        
    except Exception as e:
        logger.exception(f"Error in forecast generation: {e}")
        print(f"\nError: {e}")
        print(f"See log file for details: {log_file}")
        return 1
    
    return 0


def generate_report(summary, ai_analysis, output_dir):
    """Generate a human-readable HTML report."""
    logger = logging.getLogger(__name__)
    logger.info("Generating HTML report")
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>NVIDIA Forecast Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #333366; }}
            h2 {{ color: #333366; margin-top: 30px; }}
            .metrics {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
            .metric-card {{ 
                background-color: #f8f9fa; 
                border-radius: 5px; 
                padding: 15px; 
                margin: 10px; 
                min-width: 200px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metric-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
            .metric-name {{ font-size: 14px; color: #666; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            .image-container {{ margin: 20px 0; }}
            .image-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; }}
            .analysis {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>NVIDIA Stock Forecast with AGI Timeline Analysis</h1>
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Key Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-name">Current Price</div>
                <div class="metric-value">${summary['current_price']:.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-name">1-Month Forecast</div>
                <div class="metric-value">${summary.get('1_month_price', 'N/A')}</div>
                <div class="metric-name">{summary.get('1_month_return_pct', 0):.2f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-name">3-Month Forecast</div>
                <div class="metric-value">${summary.get('3_months_price', 'N/A')}</div>
                <div class="metric-name">{summary.get('3_months_return_pct', 0):.2f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-name">1-Year Forecast</div>
                <div class="metric-value">${summary.get('1_year_price', 'N/A')}</div>
                <div class="metric-name">{summary.get('1_year_return_pct', 0):.2f}%</div>
            </div>
        </div>
        
        <h2>AGI Probability Timeline</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-name">AGI by 2030</div>
                <div class="metric-value">{summary.get('agi_prob_2030', 0)*100:.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-name">AGI by 2035</div>
                <div class="metric-value">{summary.get('agi_prob_2035', 0)*100:.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-name">AGI by 2040</div>
                <div class="metric-value">{summary.get('agi_prob_2040', 0)*100:.1f}%</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-name">AGI by 2050</div>
                <div class="metric-value">{summary.get('agi_prob_2050', 0)*100:.1f}%</div>
            </div>
        </div>
        
        <h2>Forecast Visualization</h2>
        <div class="image-container">
            <img src="forecast_plot.png" alt="NVIDIA Stock Forecast">
        </div>
        
        <h2>AGI Probability Visualization</h2>
        <div class="image-container">
            <img src="agi_cumulative_plot.png" alt="AGI Cumulative Probability">
        </div>
        
        <h2>AGI Distribution Analysis</h2>
        <div class="image-container">
            <img src="agi_distributions_plot.png" alt="AGI Probability Distributions">
        </div>
        
        <h2>AI Landscape Analysis</h2>
        <div class="analysis">
            <h3>NVIDIA's Position in AI Hardware Market</h3>
            <p>{ai_analysis.get('nvidia_position', 'Analysis not available')}</p>
            
            <h3>Recent AI Advancements</h3>
            <p>{ai_analysis.get('ai_advancements', 'Analysis not available')}</p>
            
            <h3>AGI Timeline Estimates</h3>
            <p>{ai_analysis.get('agi_timeline', 'Analysis not available')}</p>
        </div>
        
        <h2>Model Performance</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-name">Training Score (R²)</div>
                <div class="metric-value">{summary['train_score']:.4f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-name">Testing Score (R²)</div>
                <div class="metric-value">{summary['test_score']:.4f}</div>
            </div>
        </div>
        
        <p><small>Disclaimer: This forecast is for informational purposes only and does not constitute investment advice. All forecasts have inherent limitations and uncertainties.</small></p>
    </body>
    </html>
    """
    
    # Write HTML to file
    report_file = os.path.join(output_dir, 'forecast_report.html')
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report: {report_file}")
    return report_file


if __name__ == "__main__":
    sys.exit(main())
