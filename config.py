class Config:
    # Flask configuration
    SECRET_KEY = 'your-secret-key-here'  # Change this in production
    
    # Ollama model configurations
    FILTER_MODEL = "llama3.2:3b"  # For filtering inappropriate content
    PROMPT_MODEL = "llama3.2:3b"  # For creating debate prompts
    DEBATE_MODEL = "llama3.2:3b"  # Main debate model
    MODERATOR_MODEL = "llama3.2:3b"  # For moderation
