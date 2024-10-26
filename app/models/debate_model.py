import requests
from typing import Dict, List
import logging
from json import JSONDecodeError

# Configure logging
logger = logging.getLogger(__name__)

class DebateModel:
    """
    DebateModel handles the generation of debate arguments for each side
    using the specified Ollama model.
    """

    def __init__(self, model_name: str):
        """
        Initialize the debate model with specified Ollama model.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        logger.info(f"Initialized DebateModel with {model_name}")

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
                timeout=60  # Increased from 30 to 60 seconds
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

    def generate_response(self, topic: str, context: List[str], stance: str) -> Dict:
        """
        Generate the next argument in the debate sequence.
        
        Args:
            topic (str): The debate topic
            context (List[str]): Previous debate arguments
            stance (str): Current side's position ('for' or 'against')
            
        Returns:
            Dict: Generated argument and metadata
        """
        if not topic or not stance:
            raise ValueError("Topic and stance are required")
        
        if stance not in ['for', 'against']:
            raise ValueError("Stance must be either 'for' or 'against'")

        # Create the context string with clear debate history
        context_formatted = ""
        if context:
            for i, entry in enumerate(context[-3:]):  # Look at last 3 exchanges
                if isinstance(entry, dict):
                    side = entry['side'].upper()
                    text = entry['text']
                    context_formatted += f"{side}: {text}\n"
                else:
                    context_formatted += f"Point {i+1}: {entry}\n"

        prompt = (
            f"You are participating in a casual debate about: {topic}\n"
            f"Your stance is: {stance.upper()}\n\n"
            f"Previous discussion:\n{context_formatted}\n\n"
            "Respond in a conversational way by:\n"
            "1. Using natural, casual language\n"
            "2. Keeping it brief (1-2 sentences)\n"
            "3. Making it feel like a real-time discussion\n"
            "4. Starting with phrases like 'Actually...', 'I see your point, but...', 'Let me add...'\n"
            "5. Being engaging but concise\n\n"
            "Keep your response under 30 words and make it feel like a natural conversation."
        )
        
        try:
            response = self._make_api_request(prompt)
            
            if not response.get('response'):
                raise ValueError("Empty response from model")
                
            return response
        except Exception as e:
            logger.error(f"Failed to generate debate response: {str(e)}")
            raise
