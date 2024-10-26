from typing import Dict, List
import requests
import logging
from json import JSONDecodeError

# Configure logging
logger = logging.getLogger(__name__)

class ModeratorModel:
    """
    ModeratorModel manages debate quality by monitoring arguments,
    providing interventions, and generating summaries.
    """

    def __init__(self, model_name: str):
        """
        Initialize the moderator model with specified Ollama model.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        logger.info(f"Initialized ModeratorModel with {model_name}")

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

    def evaluate_response(self, topic: str, current_argument: str, debate_history: List[str]) -> Dict:
        """
        Evaluate the quality and relevance of a debate argument.
        
        Args:
            topic (str): The debate topic
            current_argument (str): The argument to evaluate
            debate_history (List[str]): Previous debate arguments
            
        Returns:
            Dict: Evaluation results including on-topic, circular, logical ratings
        """
        if not current_argument:
            raise ValueError("No argument provided for evaluation")

        # Convert debate history items to strings if they're dictionaries
        formatted_history = []
        for entry in debate_history[-3:]:  # Get last 3 exchanges
            if isinstance(entry, dict):
                formatted_history.append(f"{entry['side'].upper()}: {entry['text']}")
            else:
                formatted_history.append(str(entry))

        context_str = '\n'.join(formatted_history)

        prompt = (
            f"As a debate moderator, evaluate the following argument in the context of the debate:\n\n"
            f"Topic: {topic}\n"
            f"Current Argument: {current_argument}\n\n"
            f"Previous Discussion:\n{context_str}\n\n"
            "Analyze and return a JSON-like response with these keys:\n"
            "1. is_on_topic (true/false)\n"
            "2. is_circular (true/false)\n"
            "3. is_logical (true/false)\n"
            "4. feedback (brief moderator feedback)"
        )
        
        try:
            response = self._make_api_request(prompt)
            return response
        except Exception as e:
            logger.error(f"Failed to evaluate response: {str(e)}")
            raise

    def generate_summary(self, topic: str, debate_history: List[str]) -> str:
        """
        Generate a concise summary of the debate's current state.
        
        Args:
            topic (str): The debate topic
            debate_history (List[str]): All previous debate arguments
            
        Returns:
            str: A summary of the debate
        """
        if not debate_history:
            raise ValueError("No debate history provided for summary")

        # Convert debate history items to strings if they're dictionaries
        formatted_history = []
        for entry in debate_history[-5:]:  # Get last 5 exchanges
            if isinstance(entry, dict):
                formatted_history.append(f"{entry['side'].upper()}: {entry['text']}")
            else:
                formatted_history.append(str(entry))

        context_str = '\n'.join(formatted_history)

        prompt = (
            f"Provide a brief, impartial summary of the following debate:\n\n"
            f"Topic: {topic}\n\n"
            f"Debate History:\n{context_str}\n\n"
            "Focus on:\n"
            "1. Key arguments from both sides\n"
            "2. Main points of contention\n"
            "3. Current state of the debate\n\n"
            "Keep the summary concise and neutral."
        )
        
        try:
            response = self._make_api_request(prompt)
            return response['response']
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            raise

    def should_intervene(self, topic: str, debate_history: List[str]) -> Dict:
        """
        Determine if moderator intervention is needed in the debate.
        
        Args:
            topic (str): The debate topic
            debate_history (List[str]): Previous debate arguments
            
        Returns:
            Dict: Decision about intervention and reason
        """
        if not debate_history:
            return {"needs_intervention": False, "reason": "Debate hasn't started yet"}

        # Convert debate history items to strings if they're dictionaries
        formatted_history = []
        for entry in debate_history[-3:]:  # Get last 3 exchanges
            if isinstance(entry, dict):
                formatted_history.append(f"{entry['side'].upper()}: {entry['text']}")
            else:
                formatted_history.append(str(entry))

        context_str = '\n'.join(formatted_history)

        prompt = (
            f"Analyze this debate and determine if moderator intervention is needed:\n\n"
            f"Topic: {topic}\n\n"
            f"Recent Discussion:\n{context_str}\n\n"
            "Check for:\n"
            "1. Off-topic discussion\n"
            "2. Circular arguments\n"
            "3. Logical fallacies\n"
            "4. Need for summary\n\n"
            "Return true if intervention needed, false if not, and include reason."
        )
        
        try:
            response = self._make_api_request(prompt)
            result = response['response'].lower()
            needs_intervention = 'true' in result
            
            return {
                "needs_intervention": needs_intervention,
                "reason": result
            }
        except Exception as e:
            logger.error(f"Failed to determine intervention need: {str(e)}")
            return {
                "needs_intervention": False,
                "reason": "Error in intervention check"
            }
