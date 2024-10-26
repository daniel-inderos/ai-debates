from typing import Dict
import requests
import logging
from json import JSONDecodeError

# Configure logging
logger = logging.getLogger(__name__)

class FilterModel:
    """
    FilterModel screens debate topics for appropriateness and
    ensures topics are suitable for constructive debate.
    """

    def __init__(self, model_name: str):
        """
        Initialize the filter model with specified Ollama model.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        logger.info(f"Initialized FilterModel with {model_name}")

    def _make_api_request(self, prompt: str) -> Dict:
        """
        Make a request to the Ollama API with error handling.
        
        Args:
            prompt (str): The prompt to send to the model
            
        Returns:
            Dict: The API response
            
        Raises:
            ConnectionError: If cannot connect to Ollama
            ValueError: If response is invalid
        """
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30  # Increased from 15 to 30 seconds
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama service")
            raise ConnectionError("AI service is unavailable")
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            raise TimeoutError("Request took too long to process")
        except JSONDecodeError:
            logger.error("Received invalid JSON response from Ollama")
            raise ValueError("Invalid response from AI service")
        except Exception as e:
            logger.error(f"Unexpected error in API request: {str(e)}")
            raise
        
    def filter_topic(self, topic: str) -> Dict[str, bool]:
        """
        Analyze a debate topic for appropriateness.
        
        Args:
            topic (str): The proposed debate topic
            
        Returns:
            Dict[str, bool]: Contains 'is_appropriate' flag and reason
            
        Raises:
            ValueError: If topic is empty or invalid
        """
        if not topic or len(topic.strip()) == 0:
            raise ValueError("Empty topic provided")

        prompt = f"""
        Analyze the following debate topic and determine if it is appropriate:
        Topic: {topic}
        
        Consider these criteria:
        1. Not harmful or promoting hate
        2. Not explicitly graphic or violent
        3. Suitable for constructive debate
        4. Not personally targeting individuals
        
        Return only 'true' if the topic is appropriate for debate, 'false' if not.
        Include a brief reason for the decision.
        """
        
        try:
            response = self._make_api_request(prompt)
            result = response['response'].strip().lower()
            
            # Parse the response
            is_appropriate = 'true' in result
            reason = result.replace('true', '').replace('false', '').strip()
            
            logger.info(f"Topic '{topic}' filtered as {'appropriate' if is_appropriate else 'inappropriate'}")
            
            return {
                "is_appropriate": is_appropriate,
                "reason": reason or "Topic analyzed for appropriateness"
            }
        except Exception as e:
            logger.error(f"Failed to filter topic: {str(e)}")
            # Default to rejecting topic on error
            return {
                "is_appropriate": False,
                "reason": "Unable to verify topic appropriateness"
            }
