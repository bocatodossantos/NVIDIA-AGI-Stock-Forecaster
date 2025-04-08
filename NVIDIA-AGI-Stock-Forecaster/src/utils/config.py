"""
Configuration utilities for the NVIDIA AGI Stock Forecaster.

This module provides functions for loading and managing configuration.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    'model_params': {
        'n_estimators': 100,
        'max_depth': None,
        'random_state': 42,
        'test_size': 0.2
    },
    'agi_model_params': {
        'expert_survey_weight': 0.5,
        'compute_trajectory_weight': 0.5,
        'economic_impact_threshold': 0.5,
        'impact_lag_years': 2,
        'impact_duration_years': 5
    },
    'data_cache_dir': 'data/raw',
    'model_save_dir': 'data/models',
    'results_dir': 'results',
    'log_level': 'INFO',
    'use_cache': True
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file and merge with defaults.
    
    Args:
        config_path (str, optional): Path to the configuration file.
            If None, default configuration will be used.
            
    Returns:
        dict: The configuration dictionary with defaults applied
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            logger.info(f"Loading configuration from {config_path}")
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                
            if user_config:
                # Merge configurations recursively
                _merge_configs(config, user_config)
                logger.info("Configuration loaded successfully")
            else:
                logger.warning(f"Config file {config_path} is empty, using defaults")
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            logger.warning("Using default configuration")
    else:
        if config_path:
            logger.warning(f"Config file {config_path} not found, using defaults")
        else:
            logger.info("No config path provided, using default configuration")
    
    return config


def _merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
    """
    Recursively merge override_config into base_config.
    
    Args:
        base_config (dict): Base configuration to update
        override_config (dict): Override configuration values
    """
    for key, value in override_config.items():
        if (
            key in base_config and 
            isinstance(base_config[key], dict) and
            isinstance(value, dict)
        ):
            _merge_configs(base_config[key], value)
        else:
            base_config[key] = value


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config (dict): Configuration to save
        config_path (str): Path to save the configuration file
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {str(e)}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    config = load_config()
    print("Default configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Example: Save config to file
    # save_config(config, 'config/example_config.yaml')
