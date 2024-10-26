# AI Debate Arena

An interactive web application where two AI models engage in structured debates on user-provided topics. Built with Python, Flask, and Ollama's llama3.2 models.

## Features

- Real-time AI debates on any topic
- Content filtering for appropriate debate topics
- Automatic stance generation for both sides
- Interactive, conversational debate style
- Moderator AI that ensures debate quality
- Automatic summaries and interventions when needed

## Prerequisites

- Python 3.9 or higher
- Ollama installed and running (https://ollama.ai/)
- llama3.2:3b model pulled from Ollama

## Quick Start

1. Install Ollama and pull the model:
   ollama pull llama3.2:3b

2. Create environment:
   conda create -n ai-debate python=3.9
   conda activate ai-debate

3. Install requirements:
   pip install -r requirements.txt

4. Run the app:
   python run.py

5. Open browser:
   http://localhost:5000

## Usage Tips

1. Enter any debate topic
2. Wait for initial stance generation
3. Watch AIs debate automatically
4. Moderator will intervene if needed
5. Get final summary at the end

## Known Issues

- First request may be slow (model loading)
- Occasional timeouts with complex topics
- Limited context window

## Future Plans

- Support for more AI models
- Streaming responses
- User feedback system
- Improved debate coherence
- Debate history saving
