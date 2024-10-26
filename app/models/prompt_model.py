from typing import Dict, Tuple
import requests
import logging
from json import JSONDecodeError

# Configure logging
logger = logging.getLogger(__name__)

class PromptModel:
    """
    PromptModel generates debate stances and system prompts for the debaters,
    helping to structure the debate and ensure quality arguments.
    """

    def __init__(self, model_name: str):
        """
        Initialize the prompt model with specified Ollama model.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        logger.info(f"Initialized PromptModel with {model_name}")

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
                timeout=60  # Increased from 20 to 60 seconds
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
    
    def generate_stances(self, topic: str) -> Tuple[str, str]:
        """
        Generate two opposing stances for a debate topic.
        
        Args:
            topic (str): The debate topic
            
        Returns:
            Tuple[str, str]: (for_stance, against_stance)
            
        Raises:
            ValueError: If topic is invalid or stance generation fails
        """
        if not topic or len(topic.strip()) == 0:
            raise ValueError("Empty topic provided")

        prompt = (
            f"You are helping set up a debate about: '{topic}'\n\n"
            "Generate exactly two opposing stances in this exact format:\n"
            "FOR: (write a one-sentence stance supporting the topic)\n"
            "AGAINST: (write a one-sentence stance opposing the topic)\n\n"
            "Make each stance clear and specific. Do not add any other text.\n"
            "Example format:\n"
            "FOR: Remote work increases productivity and work-life balance while reducing commute times and environmental impact.\n"
            "AGAINST: Traditional office work promotes better collaboration, team cohesion, and work-life separation while ensuring proper oversight."
        )
        
        try:
            response = self._make_api_request(prompt)
            result = response['response']
            
            # More robust parsing
            for_stance = ""
            against_stance = ""
            
            # Try multiple parsing methods
            lines = [line.strip() for line in result.split('\n') if line.strip()]
            for line in lines:
                line = line.strip()
                if line.upper().startswith('FOR:'):
                    for_stance = line[4:].strip()
                elif line.upper().startswith('AGAINST:'):
                    against_stance = line[8:].strip()
                # Backup parsing if colons are missing
                elif line.upper().startswith('FOR '):
                    for_stance = line[4:].strip()
                elif line.upper().startswith('AGAINST '):
                    against_stance = line[8:].strip()
            
            # If still empty, try to split the response differently
            if not for_stance and not against_stance and len(lines) >= 2:
                for_stance = lines[0].strip()
                against_stance = lines[1].strip()
            
            # Clean up the stances
            for_stance = for_stance.replace('FOR:', '').replace('FOR', '').strip()
            against_stance = against_stance.replace('AGAINST:', '').replace('AGAINST', '').strip()
            
            # Validate stances
            if len(for_stance) < 5 or len(against_stance) < 5:
                logger.warning(f"Generated stances too short, retrying - Raw response: {result}")
                # Retry once with a simpler prompt
                return self._retry_generate_stances(topic)
            
            logger.info(f"Successfully generated stances - For: '{for_stance}', Against: '{against_stance}'")
            return for_stance, against_stance
            
        except Exception as e:
            logger.error(f"Failed to generate stances for topic '{topic}': {str(e)}")
            raise ValueError(f"Failed to generate debate stances: {str(e)}")

    def _retry_generate_stances(self, topic: str) -> Tuple[str, str]:
        """Fallback method with simpler prompt for generating stances."""
        simple_prompt = (
            f"Topic: {topic}\n\n"
            "1. Write one sentence supporting this topic.\n"
            "2. Write one sentence opposing this topic.\n"
            "Be clear and specific."
        )
        
        try:
            response = self._make_api_request(simple_prompt)
            lines = response['response'].split('\n')
            lines = [line.strip() for line in lines if line.strip()]
            
            if len(lines) >= 2:
                return lines[0], lines[1]
            else:
                raise ValueError("Could not generate valid stances even with retry")
        except Exception as e:
            logger.error(f"Retry failed for topic '{topic}': {str(e)}")
            raise ValueError("Failed to generate debate stances even with retry")

    def generate_system_prompt(self, stance: str, topic: str) -> str:
        """
        Create a system prompt for an AI debater.
        
        Args:
            stance (str): The position the AI should argue for
            topic (str): The debate topic
            
        Returns:
            str: Generated system prompt
            
        Raises:
            ValueError: If stance or topic is invalid
        """
        if not stance or not topic:
            raise ValueError("Stance and topic are required")

        prompt = f"""
        Create a system prompt for an AI debater that will argue {stance} on the topic: "{topic}"
        
        The system prompt should include:
        1. Clear definition of the AI's role and perspective
        2. Guidelines for maintaining respectful discourse
        3. Requirements for using logic and evidence
        4. Instructions for keeping responses focused and concise
        5. Strategies for addressing counter-arguments
        
        Return only the system prompt, no explanations.
        Make it clear and actionable.
        """
        
        try:
            response = self._make_api_request(prompt)
            system_prompt = response['response'].strip()
            
            # Validate system prompt
            if len(system_prompt) < 50:  # Basic validation
                raise ValueError("Generated system prompt is too short")
                
            logger.info(f"Successfully generated system prompt for stance: {stance}")
            return system_prompt
            
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {str(e)}")
            raise
