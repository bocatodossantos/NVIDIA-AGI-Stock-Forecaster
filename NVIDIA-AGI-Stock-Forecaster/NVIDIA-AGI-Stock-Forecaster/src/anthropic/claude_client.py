
"""
Claude API client for interacting with Anthropic's Claude AI assistant.

This module provides a simple client for calling the Anthropic Claude API
to get AI analysis of various topics relevant to the NVIDIA forecast.
"""

import requests
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AnthropicClient:
    """Simple client for calling Anthropic Claude API"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize the Anthropic client.exit
        
        Args:
            api_key (str): Your Anthropic API key
            model (str): The Claude model to use
        """
        self.api_key = api_key
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = model
        logger.info(f"Initialized Anthropic client using model: {model}")
    
    def ask(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Ask Claude a question and get a response.
        
        Args:
            prompt (str): The question or prompt to send to Claude
            max_tokens (int): Maximum number of tokens in the response
            temperature (float): Controls randomness (0.0 to 1.0)
            
        Returns:
            str: Claude's response
            
        Raises:
            Exception: If the API call fails
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        logger.debug(f"Sending request to Anthropic API with prompt: {prompt[:100]}...")
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Log the beginning of the response for debugging
            response_text = result["content"][0]["text"]
            logger.debug(f"Received response from Claude: {response_text[:100]}...")
            
            return response_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Anthropic API: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise
    
    def batch_analyze(self, topics: Dict[str, str]) -> Dict[str, str]:
        """
        Analyze multiple topics in batch.
        
        Args:
            topics: Dictionary mapping topic names to prompts
            
        Returns:
            Dictionary mapping topic names to Claude's responses
        """
        results = {}
        for topic_name, prompt in topics.items():
            logger.info(f"Analyzing topic: {topic_name}")
            try:
                results[topic_name] = self.ask(prompt)
            except Exception as e:
                logger.error(f"Error analyzing topic {topic_name}: {e}")
                results[topic_name] = f"Error: {str(e)}"
        return results


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Replace with your actual API key
    API_KEY = "YOUR_ANTHROPIC_API_KEY"
    
    # Create client
    client = AnthropicClient(API_KEY)
    
    # Test query
    try:
        response = client.ask("What is NVIDIA's current position in the AI hardware market? Keep it brief.")
        print("\nResponse from Claude:")
        print(response)
    except Exception as e:
        print(f"Error: {e}")