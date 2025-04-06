from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import uuid
import datetime
import threading
import time
import gc
import os
import logging
from logging.handlers import RotatingFileHandler
import platform
from werkzeug.middleware.proxy_fix import ProxyFix
import webbrowser
import subprocess

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('logs/app.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app)  # Enable CORS for all routes

# Session memory
chat_sessions = {}
session_last_active = {}

# Model and tokenizer
tokenizer = None
model = None
model_loaded = False
model_load_time = None

# Constants - Using environment variables with defaults
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# Simplified chat template that just includes the bare necessary structure
CHAT_TEMPLATE = os.environ.get("CHAT_TEMPLATE", "<|system|>\n{system_message}\n<|user|>\n{user_message}\n<|assistant|>")
# Simplified system prompt without markdown instructions
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a concise and friendly chatbot. Answer directly and briefly.")
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "1800"))  # 30 minutes
MAX_SESSION_LENGTH = int(os.environ.get("MAX_SESSION_LENGTH", "10"))
MEMORY_CLEANUP_INTERVAL = int(os.environ.get("MEMORY_CLEANUP_INTERVAL", "300"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))  # Reduced for better coherence
MAX_CONVERSATION_HISTORY = int(os.environ.get("MAX_CONVERSATION_HISTORY", "20"))

def get_system_info():
    """Get system information including GPU status"""
    info = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME,
        "active_sessions": len(chat_sessions),
        "gpu_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB"
        })
    
    return info

def validate_environment():
    """Validate and log environment configuration"""
    env_vars = {
        'MODEL_NAME': MODEL_NAME,
        'SESSION_TIMEOUT': SESSION_TIMEOUT,
        'MAX_SESSION_LENGTH': MAX_SESSION_LENGTH,
        'MEMORY_CLEANUP_INTERVAL': MEMORY_CLEANUP_INTERVAL,
        'MAX_TOKENS': MAX_TOKENS,
        'MAX_CONVERSATION_HISTORY': MAX_CONVERSATION_HISTORY
    }
    
    logger.info('Environment configuration:')
    for var, value in env_vars.items():
        logger.info(f'{var}: {value}')

def setup_model():
    """Initialize the model and tokenizer"""
    global model, tokenizer, model_loaded, model_load_time
    logger.info(f"Setting up {MODEL_NAME} model...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = 'right'

        if torch.cuda.is_available():
            logger.info("GPU available. Loading model to GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16  # Use half precision for GPU
            ).to("cuda")
        else:
            logger.info("GPU not available. Loading model to CPU...")
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to("cpu")

        model_loaded = True
        model_load_time = datetime.datetime.now()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        model_loaded = False

def format_chat_history(session_id):
    """Format the chat history for the model with better context management"""
    history = chat_sessions.get(session_id, [])
    if not history:
        return None

    # Make sure we only take the most recent messages to avoid context overflow
    context_window = history[-MAX_CONVERSATION_HISTORY:]
    
    formatted_history = []
    
    # Start with system message
    formatted_history.append(f"<|system|>\n{SYSTEM_PROMPT}")
    
    # Add conversation history
    for msg in context_window:
        if msg['role'] == 'user':
            formatted_history.append(f"<|user|>\n{msg['content']}")
        else:
            formatted_history.append(f"<|assistant|>\n{msg['content']}")

    # Only add the assistant prompt marker if the last message was from a user
    if history and history[-1]['role'] == 'user':
        return ''.join(formatted_history) + "\n<|assistant|>"
    else:
        return ''.join(formatted_history)

def generate_response(prompt, temperature=0.3, max_tokens=MAX_TOKENS):
    """Generate model response with improved parameters for more coherent outputs"""
    if not model_loaded:
        return "Model is not ready yet. Please try again shortly."
    
    if not prompt:
        return "Please provide a message to respond to."

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,  # Lower temperature for more deterministic outputs
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.2,  # Increased repetition penalty
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        return "I encountered an error while processing your request. Please try again."

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with improved session management"""
    if not model_loaded:
        return jsonify({
            'status': 'error',
            'message': 'Model is still loading. Please wait.'
        }), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Invalid JSON data'
            }), 400
            
        session_id = data.get('session_id') or str(uuid.uuid4())
        user_message = data.get('message', '').strip()

        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'Empty message'
            }), 400

        # Initialize session if needed
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
            
        # Add user message to session
        chat_sessions[session_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

        # Generate response
        prompt = format_chat_history(session_id)
        response = generate_response(prompt)

        # Add assistant response to session
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.datetime.utcnow().isoformat()
        })

        # Update session timestamp
        session_last_active[session_id] = datetime.datetime.now()

        return jsonify({
            'status': 'success',
            'response': response,
            'session_id': session_id
        })
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def status():
    """Get detailed server status"""
    try:
        status_info = get_system_info()
        return jsonify(status_info)
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear a chat session"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        if session_id in chat_sessions:
            chat_sessions.pop(session_id, None)
            session_last_active.pop(session_id, None)
            return jsonify({
                'status': 'success',
                'message': f'Session {session_id} cleared.'
            })
        return jsonify({
            'status': 'success',
            'message': 'Session not found or already cleared.'
        })
    except Exception as e:
        logger.error(f"Error in clear_session endpoint: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/')
def serve_index():
    """Serve the index.html file"""
    return send_from_directory('.', 'index.html')

def cleanup_sessions():
    """Periodically clean up expired sessions and manage memory"""
    while True:
        try:
            now = datetime.datetime.now()
            expired_sessions = [
                sid for sid, last_active in session_last_active.items()
                if (now - last_active).total_seconds() > SESSION_TIMEOUT
            ]
            
            for sid in expired_sessions:
                chat_sessions.pop(sid, None)
                session_last_active.pop(sid, None)
                logger.info(f"Session {sid} cleaned up due to inactivity")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            time.sleep(MEMORY_CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup_sessions: {e}", exc_info=True)
            time.sleep(60)

if __name__ == '__main__':
    # Validate environment
    validate_environment()
    
    # Start model loading in a separate thread
    threading.Thread(target=setup_model, daemon=True).start()
    
    # Start session cleanup in a separate thread
    threading.Thread(target=cleanup_sessions, daemon=True).start()
    
    # Start Flask app and open web browser.
    port = int(os.environ.get("PORT", 5000))
    logger = logging.getLogger(__name__)

    def is_wsl():
        try:
            with open("/proc/version", "r") as f:
                return "microsoft" in f.read().lower()
        except FileNotFoundError:
            return False

    def open_browser(port):
        url = f"http://localhost:{port}"

        # Try opening with PowerShell only if on Windows
        if platform.system() == "Windows" and not is_wsl():
            try:
                subprocess.run(["powershell.exe", "start", url], check=True)
                return
            except Exception as e:
                logger.warning(f"Could not open browser with PowerShell: {e}")

        #Fallback to Python's webbrowser
        try:
            webbrowser.open(url)
        except Exception as e:
            logger.warning(f"Could not open browser with webbrowser: {e}")



    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
