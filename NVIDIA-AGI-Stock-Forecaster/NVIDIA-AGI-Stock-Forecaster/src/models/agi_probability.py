"""
AGI Probability Model for estimating AGI development timelines.

This module implements a sophisticated model for estimating AGI timeline 
probabilities based on multiple methodologies and expert forecasts.
"""

import numpy as np
import pandas as pd
from scipy.stats import lognorm, weibull_min, norm
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AGIProbabilityModel:
    """
    A model for estimating AGI timeline probabilities based on multiple
    methodologies and expert forecasts.
    """
    
    def __init__(self, anthropic_client=None, current_year=None):
        """
        Initialize the AGI probability model.
        
        Args:
            anthropic_client: An instance of AnthropicClient for analyzing AI developments
            current_year (int, optional): The current year. If None, uses the current year.
        """
        # Set current year
        self.current_year = current_year or datetime.now().year
        logger.info(f"Initializing AGI Probability Model (base year: {self.current_year})")
        
        # Parameters
        self.compute_doubling_time = 2.5  # years
        self.expert_survey_data = self._load_expert_surveys()
        self.scaling_laws_params = self._load_scaling_laws_params()
        self.economic_impact_threshold = 0.5  # threshold for economic impact
        self.anthropic_client = anthropic_client
    
    def _load_expert_surveys(self):
        """
        Load and process expert survey data on AGI timelines.
        In a real implementation, this would pull from actual survey datasets.
        
        Returns:
            pd.DataFrame: Expert survey data
        """
        logger.debug("Loading expert survey data")
        
        # Simulated expert survey data
        # Format: [source, median_year, confidence_interval_low, confidence_interval_high]
        return pd.DataFrame([
            ['AI Impacts 2022', 2045, 2035, 2065],
            ['Grace et al. 2020', 2050, 2036, 2075],
            ['Zhang et al. 2022', 2040, 2030, 2060],
            ['Metaculus Community', 2035, 2028, 2055],
            ['AI Leaders Survey', 2032, 2027, 2045],
            ['ML Researchers', 2039, 2029, 2058]
        ], columns=['source', 'median_year', 'ci_low', 'ci_high'])
    
    def _load_scaling_laws_params(self):
        """
        Load parameters for ML scaling law analysis.
        Based on papers like Kaplan et al. and follow-up work on scaling laws.
        
        Returns:
            dict: Scaling law parameters
        """
        logger.debug("Loading scaling laws parameters")
        
        return {
            'compute_current': 1e26,  # current AI training compute in FLOP
            'compute_agi_estimate': 1e30,  # estimated compute for AGI in FLOP
            'compute_growth_rate': 1.3,  # annual multiplier in compute
            'parameter_scaling_exponent': 0.75,  # performance scaling with parameters
            'data_scaling_exponent': 0.5,  # performance scaling with data
            'effective_horizon': 15  # years over which we can reasonably extrapolate
        }
    
    def get_claude_analysis(self, topic):
        """
        Get Claude's analysis on a specific AGI-related topic if client is available.
        
        Args:
            topic (str): The topic to analyze ('nvidia_position', 'ai_advancements', 
                        or 'agi_timeline')
                        
        Returns:
            str: Analysis from Claude
        """
        if not self.anthropic_client:
            return "Claude analysis not available (no API client provided)"
        
        logger.info(f"Getting Claude analysis on topic: {topic}")
        
        prompts = {
            "nvidia_position": """
            As an AI expert, provide a concise, factual analysis of NVIDIA's current position in the AI hardware market.
            Focus on market share, technological advantages, and potential challenges.
            Include quantitative metrics where possible. Limit your response to 3-4 paragraphs.
            """,
            
            "ai_advancements": """
            Provide a concise, factual summary of the most significant AI advancements 
            in the last 6-12 months. Focus on breakthroughs in model architecture, 
            capabilities, and computational efficiency. Mention only developments that 
            have been peer-reviewed or widely acknowledged by experts.
            Limit your response to 3-4 paragraphs.
            """
        }
        
        if topic not in prompts:
            return f"Unknown topic: {topic}"
        
        try:
            return self.anthropic_client.ask(prompts[topic])
        except Exception as e:
            logger.error(f"Error getting Claude analysis: {e}")
            return f"Error analyzing {topic}: {str(e)}"
    
    def _expert_distribution(self):
        """
        Create a distribution based on expert surveys.
        Returns a function representing the PDF of AGI arrival time.
        
        Returns:
            callable: Probability density function for AGI arrival time
        """
        logger.debug("Creating expert-based probability distribution")
        
        # Calculate the mean and std from the survey data
        years = self.expert_survey_data['median_year'].values
        ci_ranges = self.expert_survey_data['ci_high'] - self.expert_survey_data['ci_low']
        
        # Convert confidence intervals to standard deviations (assuming 95% CI)
        stds = ci_ranges / (2 * 1.96)
        
        # Create a mixture distribution from all surveys
        def mixture_pdf(x):
            pdfs = np.zeros_like(x, dtype=float)
            for year, std in zip(years, stds):
                pdfs += norm.pdf(x, loc=year, scale=std)
            return pdfs / len(years)
        
        return mixture_pdf
    
    def _compute_trajectory_distribution(self):
        """
        Create a distribution based on compute scaling trajectories.
        Returns a function representing the PDF of AGI arrival time.
        
        Returns:
            callable: Probability density function based on compute trends
        """
        logger.debug("Creating compute trajectory-based probability distribution")
        
        p = self.scaling_laws_params
        
        # Years to reach AGI compute threshold
        log_compute_ratio = np.log(p['compute_agi_estimate'] / p['compute_current'])
        log_annual_growth = np.log(p['compute_growth_rate'])
        mean_years = log_compute_ratio / log_annual_growth
        
        # Add uncertainty - higher for longer forecasts
        std_years = mean_years * 0.3
        
        # Lognormal distribution to reflect uncertainty increasing with time
        shape = np.sqrt(np.log(1 + (std_years/mean_years)**2))
        scale = np.log(mean_years) - 0.5 * shape**2
        
        # Create the PDF
        def compute_pdf(x):
            # Shift to make the distribution start at the current year
            shifted_x = x - self.current_year
            # Return PDF values for years >= current year
            return np.where(shifted_x > 0, 
                            lognorm.pdf(shifted_x, s=shape, scale=np.exp(scale)), 
                            0)
        
        return compute_pdf
    
    def run_simplified_model(self):
        """
        Run a simplified model without PyMC to combine evidence sources.
        
        Returns:
            tuple: (posterior_mean, posterior_std) of the AGI timeline distribution
        """
        logger.info("Running simplified AGI probability model")
        
        # Create year range
        years = np.linspace(self.current_year, self.current_year + 40, 1000)
        
        # Get the PDFs
        expert_pdf = self._expert_distribution()
        compute_pdf = self._compute_trajectory_distribution()
        
        # Combine PDFs with equal weighting
        combined_pdf = lambda x: 0.5 * expert_pdf(x) + 0.5 * compute_pdf(x)
        
        # Calculate combined PDF values
        combined_values = combined_pdf(years)
        
        # Normalize to ensure it integrates to 1
        norm_factor = np.trapz(combined_values, years)
        combined_values = combined_values / norm_factor
        
        # Find the mean (expected value) of the distribution
        self.posterior_mean = np.trapz(years * combined_values, years)
        
        # Find the variance and standard deviation
        variance = np.trapz((years - self.posterior_mean)**2 * combined_values, years)
        self.posterior_std = np.sqrt(variance)
        
        logger.info(f"AGI Timeline Estimate: Mean year = {self.posterior_mean:.1f}, Std = {self.posterior_std:.1f} years")
        
        # Store the combined PDF for later use
        self.years = years
        self.pdf_values = combined_values
        
        return self.posterior_mean, self.posterior_std
    
    def _calculate_cumulative_probability(self, year):
        """
        Calculate the cumulative probability of AGI by a given year.
        
        Args:
            year (int): The year to calculate probability for
            
        Returns:
            float: Probability of AGI by the given year
        """
        if not hasattr(self, 'posterior_mean'):
            self.run_simplified_model()
            
        # Calculate probability based on normal approximation
        prob = norm.cdf(year, loc=self.posterior_mean, scale=self.posterior_std)
        return prob
    
    def get_agi_probability_by_year(self, years):
        """
        Get the probability of AGI for each year in the provided range.
        
        Args:
            years: Array or list of years to evaluate
            
        Returns:
            pd.DataFrame: DataFrame with years and probabilities
        """
        if not hasattr(self, 'posterior_mean'):
            self.run_simplified_model()
            
        logger.debug(f"Calculating AGI probabilities for years: {min(years)} to {max(years)}")
            
        # Calculate probability for each year
        probs = [self._calculate_cumulative_probability(year) for year in years]
        
        # Create DataFrame
        result = pd.DataFrame({
            'year': years,
            'cumulative_probability': probs
        })
        
        # Add year-on-year probability (probability AGI arrives in that specific year)
        result['yearly_probability'] = result['cumulative_probability'].diff().fillna(
            result['cumulative_probability'].iloc[0])
        
        return result
    
    def get_stock_impact_factor(self, years, impact_lag=2, impact_duration=5):
        """
        Translate AGI probabilities into stock impact factors.
        
        This models how the market might price in AGI expectations.
        
        Args:
            years: Years to evaluate
            impact_lag: Years of lag before AGI developments affect stock prices
            impact_duration: Years over which the impact is spread
            
        Returns:
            pd.DataFrame: DataFrame with years and impact factors
        """
        logger.debug(f"Calculating stock impact factors for years: {min(years)} to {max(years)}")
        
        # Get AGI probabilities
        agi_probs = self.get_agi_probability_by_year(years)
        
        # Calculate the financial impact factor
        # This models market reaction to AGI development probabilities
        impact_factors = []
        
        for i, year in enumerate(years):
            # Current year AGI probability
            current_prob = agi_probs.loc[agi_probs['year'] == year, 'cumulative_probability'].iloc[0]
            
            # Calculate impact factor based on:
            # 1. The probability AGI will arrive in the next few years
            future_years = [y for y in years if year < y <= year + impact_duration]
            future_probs = agi_probs.loc[agi_probs['year'].isin(future_years), 'yearly_probability']
            
            # Weight by time proximity - closer AGI arrival has more impact
            weights = np.array([(1 - (y - year) / impact_duration) for y in future_years])
            weighted_future_impact = np.sum(future_probs.values * weights) if len(future_years) > 0 else 0
            
            # 2. Current AGI expectations
            current_impact = current_prob * self.economic_impact_threshold
            
            # 3. Discount for impact lag
            discount_factor = 1 / (1 + 0.1) ** impact_lag  # 10% discount rate
            
            # Combine factors
            impact_factor = (current_impact + weighted_future_impact) * discount_factor
            impact_factors.append(impact_factor)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'year': years,
            'agi_probability': agi_probs['cumulative_probability'].values,
            'stock_impact_factor': impact_factors
        })
        
        return result
    
    def plot_distributions(self, save_path=None):
        """
        Plot the different probability distributions and the combined posterior.
        
        Args:
            save_path (str, optional): If provided, saves the plot to this path
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        import matplotlib.pyplot as plt
        logger.info("Plotting AGI probability distributions")
        
        # Create year range
        years = np.linspace(self.current_year, self.current_year + 40, 1000)
        
        # Get the PDFs
        expert_pdf = self._expert_distribution()
        compute_pdf = self._compute_trajectory_distribution()
        
        # Calculate values
        expert_values = expert_pdf(years)
        compute_values = compute_pdf(years)
        
        # Posterior distribution from simplified model
        if hasattr(self, 'posterior_mean'):
            posterior_values = norm.pdf(years, loc=self.posterior_mean, scale=self.posterior_std)
        else:
            self.run_simplified_model()
            posterior_values = norm.pdf(years, loc=self.posterior_mean, scale=self.posterior_std)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        plt.plot(years, expert_values, label='Expert Surveys', linewidth=2)
        plt.plot(years, compute_values, label='Compute Trajectory', linewidth=2)
        plt.plot(years, posterior_values, 'k--', linewidth=3, label='Combined Distribution')
        
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Probability Density', fontsize=14)
        plt.title('AGI Timing Probability Distributions by Methodology', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        
        # Add vertical lines for key percentiles
        percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        percentile_years = norm.ppf(percentiles, loc=self.posterior_mean, scale=self.posterior_std)
        
        colors = ['r', 'orange', 'g', 'orange', 'r']
        labels = ['10%', '25%', '50%', '75%', '90%']
        
        for year, color, label in zip(percentile_years, colors, labels):
            plt.axvline(year, color=color, linestyle=':', alpha=0.7)
            plt.text(year+0.5, plt.gca().get_ylim()[1]*0.9, f'{label}: {year:.1f}', 
                     rotation=90, color=color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved distribution plot to {save_path}")
            
        return plt.gcf()
    
    def plot_cumulative_probability(self, save_path=None):
        """
        Plot the cumulative probability of AGI arrival over time.
        
        Args:
            save_path (str, optional): If provided, saves the plot to this path
            
        Returns:
            matplotlib.figure.Figure: The plot figure
        """
        import matplotlib.pyplot as plt
        
        if not hasattr(self, 'posterior_mean'):
            self.run_simplified_model()
            
        logger.info("Plotting cumulative AGI probability")
            
        # Create year range
        years = np.arange(self.current_year, self.current_year + 40)
        
        # Get probabilities
        probs = self.get_agi_probability_by_year(years)
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        plt.plot(probs['year'], probs['cumulative_probability'], 'b-', linewidth=3)
        
        # Add markers for key thresholds
        thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
        
        for threshold in thresholds:
            # Find closest year
            idx = np.abs(probs['cumulative_probability'] - threshold).argmin()
            year = probs['year'].iloc[idx]
            actual_prob = probs['cumulative_probability'].iloc[idx]
            
            plt.plot(year, actual_prob, 'ro', markersize=8)
            plt.text(year+0.5, actual_prob-0.02, f'{year:.0f}', fontsize=12)
        
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Cumulative Probability', fontsize=14)
        plt.title('Cumulative Probability of AGI Arrival', fontsize=16)
        plt.grid(True)
        plt.ylim(0, 1)
        
        # Add annotations
        plt.annotate('Probability exceeds 10%', xy=(probs['year'].iloc[np.where(probs['cumulative_probability'] >= 0.1)[0][0]], 0.1),
                    xytext=(probs['year'].iloc[0]+5, 0.2), arrowprops=dict(arrowstyle='->'))
        
        plt.annotate('50% probability (median)', xy=(probs['year'].iloc[np.where(probs['cumulative_probability'] >= 0.5)[0][0]], 0.5),
                    xytext=(probs['year'].iloc[0]+10, 0.6), arrowprops=dict(arrowstyle='->'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            logger.info(f"Saved cumulative probability plot to {save_path}")
            
        return plt.gcf()


if __name__ == "__main__":
    # Simple example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    model = AGIProbabilityModel()
    model.run_simplified_model()
    
    # Print some probabilities
    years = [2030, 2035, 2040, 2050, 2060]
    for year in years:
        prob = model._calculate_cumulative_probability(year)
        print(f"Probability of AGI by {year}: {prob*100:.1f}%")
        
    # Plot distributions
    model.plot_distributions()
    model.plot_cumulative_probability()
    
    import matplotlib.pyplot as plt
    plt.show() paragraphs.
            """,
            
            "agi_timeline": """
            Based solely on published research and expert surveys, what are the current 
            predictions for AGI timeline? Provide a balanced view including both conservative
            and accelerated timeline estimates. Include key factors that might affect these
            timelines. Do not speculate beyond what's in the scientific literature.
            Limit your response to 3-4