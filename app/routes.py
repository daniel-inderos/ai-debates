from flask import Blueprint, render_template, request, jsonify, current_app
from app.models.filter_model import FilterModel
from app.models.prompt_model import PromptModel
from app.models.debate_model import DebateModel
from app.models.moderator_model import ModeratorModel
import logging
import requests  # Add this import
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)

# Error handling decorator
def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to Ollama service")
            return jsonify({
                'status': 'error',
                'message': 'AI service is currently unavailable. Please ensure Ollama is running.'
            }), 503
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': 'An unexpected error occurred. Please try again.'
            }), 500
    return decorated_function

# Initialize models with error handling
def get_models():
    try:
        filter_model = FilterModel(current_app.config['FILTER_MODEL'])
        prompt_model = PromptModel(current_app.config['PROMPT_MODEL'])
        debate_model_for = DebateModel(current_app.config['DEBATE_MODEL'])
        debate_model_against = DebateModel(current_app.config['DEBATE_MODEL'])
        moderator_model = ModeratorModel(current_app.config['MODERATOR_MODEL'])
        return filter_model, prompt_model, debate_model_for, debate_model_against, moderator_model
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        raise

@main.route('/')
def index():
    return render_template('debate.html')

@main.route('/start_debate', methods=['POST'])
@handle_errors
def start_debate():
    """
    Initialize a new debate session:
    1. Validate and filter the topic
    2. Generate opposing stances
    3. Create system prompts for debaters
    """
    topic = request.json.get('topic')
    if not topic or len(topic.strip()) == 0:
        return jsonify({
            'status': 'error',
            'message': 'Please provide a valid debate topic'
        }), 400

    # Get model instances
    filter_model, prompt_model, *_ = get_models()

    # Step 1: Filter the topic
    try:
        filter_result = filter_model.filter_topic(topic)
        if not filter_result['is_appropriate']:
            return jsonify({
                'status': 'error',
                'message': 'Topic is not appropriate for debate. Please choose another topic.'
            }), 400
    except Exception as e:
        logger.error(f"Topic filtering failed: {str(e)}")
        raise

    # Step 2: Generate debate stances
    try:
        for_stance, against_stance = prompt_model.generate_stances(topic)
        if not for_stance or not against_stance:
            raise ValueError("Failed to generate valid debate stances")
        
        # Step 3: Generate system prompts
        for_system_prompt = prompt_model.generate_system_prompt(for_stance, topic)
        against_system_prompt = prompt_model.generate_system_prompt(against_stance, topic)

        return jsonify({
            'status': 'success',
            'data': {
                'topic': topic,
                'for_stance': for_stance,
                'against_stance': against_stance,
                'for_system_prompt': for_system_prompt,
                'against_system_prompt': against_system_prompt
            }
        })
    except Exception as e:
        logger.error(f"Debate initialization failed: {str(e)}")
        raise

@main.route('/debate_round', methods=['POST'])
@handle_errors
def debate_round():
    """
    Handle a single round of debate where AIs engage in back-and-forth discussion.
    The AIs will:
    1. Generate their arguments considering previous points
    2. Have the moderator evaluate and guide the discussion
    3. Build upon each other's points
    """
    data = request.json
    required_fields = ['topic', 'current_side']
    if not all(field in data for field in required_fields):
        return jsonify({
            'status': 'error',
            'message': 'Missing required debate information'
        }), 400

    topic = data.get('topic')
    debate_history = data.get('debate_history', [])
    current_side = data.get('current_side')

    if current_side not in ['for', 'against']:
        return jsonify({
            'status': 'error',
            'message': 'Invalid debate side specified'
        }), 400

    # Get model instances
    _, _, debate_model_for, debate_model_against, moderator_model = get_models()

    try:
        # Generate argument for current side
        current_model = debate_model_for if current_side == 'for' else debate_model_against
        response = current_model.generate_response(
            topic=topic,
            context=debate_history,
            stance=current_side
        )
        
        current_argument = response['response']
        
        # Add the argument to debate history
        if debate_history is None:
            debate_history = []
        debate_history.append({
            'side': current_side,
            'text': current_argument
        })

        # Check if moderator should intervene
        moderator_check = moderator_model.should_intervene(topic, [h['text'] for h in debate_history])
        
        if moderator_check['needs_intervention']:
            # Get moderator's summary and guidance
            summary = moderator_model.generate_summary(topic, [h['text'] for h in debate_history])
            
            return jsonify({
                'status': 'moderator_intervention',
                'argument': current_argument,
                'message': moderator_check['reason'],
                'summary': summary,
                'next_side': 'for' if current_side == 'against' else 'against'
            })
        
        # Evaluate the argument
        evaluation = moderator_model.evaluate_response(
            topic=topic,
            current_argument=current_argument,
            debate_history=[h['text'] for h in debate_history]
        )
        
        return jsonify({
            'status': 'success',
            'argument': current_argument,
            'evaluation': evaluation,
            'next_side': 'for' if current_side == 'against' else 'against',
            'debate_history': debate_history
        })

    except Exception as e:
        logger.error(f"Debate round generation failed: {str(e)}")
        raise

@main.route('/end_debate', methods=['POST'])
@handle_errors
def end_debate():
    """Generate final debate summary and conclude the session"""
    data = request.json
    if 'topic' not in data or 'debate_history' not in data:
        return jsonify({
            'status': 'error',
            'message': 'Missing required debate information for summary'
        }), 400

    try:
        _, _, _, _, moderator_model = get_models()
        final_summary = moderator_model.generate_summary(
            data['topic'], 
            data['debate_history']
        )
        
        return jsonify({
            'status': 'success',
            'final_summary': final_summary
        })
    except Exception as e:
        logger.error(f"Debate conclusion failed: {str(e)}")
        raise
