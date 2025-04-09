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
import sys
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

# Thread safety
session_lock = threading.Lock()

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
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT", 
    "You are a knowledgeable and friendly chatbot. Provide helpful, informative, and engaging responses."
)
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "1800"))  # 30 minutes
MAX_SESSION_LENGTH = int(os.environ.get("MAX_SESSION_LENGTH", "10"))
MEMORY_CLEANUP_INTERVAL = int(os.environ.get("MEMORY_CLEANUP_INTERVAL", "300"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))  # Increased from 256 to 1024
MAX_CONVERSATION_HISTORY = int(os.environ.get("MAX_CONVERSATION_HISTORY", "20"))

def get_system_info():
    """Get system information including GPU status and memory usage"""
    info = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "model_loaded": model_loaded,
        "model_name": MODEL_NAME,
        "active_sessions": len(chat_sessions),
        "gpu_available": torch.cuda.is_available(),
        "system": platform.system(),
        "python_version": platform.python_version(),
        "memory_cleanup_interval": MEMORY_CLEANUP_INTERVAL,
        "session_timeout": SESSION_TIMEOUT,
        "max_tokens": MAX_TOKENS,
        "load_time": model_load_time.isoformat() if model_load_time else None
    }
    
    # Add GPU info if available
    if torch.cuda.is_available():
        info.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**2:.2f} MB",
            "gpu_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**2:.2f} MB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        })
    
    # Add CPU info
    try:
        import psutil
        info.update({
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available": f"{psutil.virtual_memory().available / 1024**3:.2f} GB"
        })
    except ImportError:
        info["psutil_available"] = False
    
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

        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU available. Loading model to GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,  # Use half precision for GPU
                low_cpu_mem_usage=True      # Helps with memory management
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
    """Format the chat history for the model with optimization for longer responses"""
    with session_lock:
        history = chat_sessions.get(session_id, [])
    
    if not history:
        return None

    # Make sure we only take the most recent messages to avoid context overflow
    context_window = history[-MAX_CONVERSATION_HISTORY:]
    
    formatted_history = []
    
    # Start with system message - enhanced for more detailed responses
    formatted_history.append(f"<|system|>\n{SYSTEM_PROMPT} Aim to provide comprehensive and thoughtful answers when appropriate.")
    
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

def generate_response(prompt, temperature=0.3, max_tokens=MAX_TOKENS, response_length="medium"):
    """Generate model response with parameters optimized for longer, more detailed outputs
    
    Parameters:
        prompt (str): The formatted prompt to send to the model
        temperature (float): Controls randomness, lower = more deterministic
        max_tokens (int): Maximum number of tokens to generate
        response_length (str): Controls response length - 'short', 'medium', 'long'
    """
    if not model_loaded:
        return "Model is not ready yet. Please try again shortly."
    
    if not prompt:
        return "Please provide a message to respond to."

    try:
        # Memory management before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Adjust parameters based on desired response length
        if response_length == "short":
            actual_max_tokens = min(256, max_tokens)
            actual_temp = max(0.2, temperature - 0.1)
            topp = 0.9
            topk = 30
            repetition_penalty = 1.1
        elif response_length == "long":
            actual_max_tokens = max_tokens  # Use full max_tokens
            actual_temp = min(0.8, temperature + 0.1)  # Slightly higher temperature for creativity
            topp = 0.95
            topk = 50
            repetition_penalty = 1.3  # Higher repetition penalty for longer responses
        else:  # medium (default)
            actual_max_tokens = int(max_tokens * 0.7)
            actual_temp = temperature
            topp = 0.92
            topk = 40
            repetition_penalty = 1.2
            
        # Adjust prompt to encourage longer responses for "long" setting
        adjusted_prompt = prompt
        if response_length == "long":
            # Extract the system message part and add instructions for more detailed responses
            system_start = adjusted_prompt.find("<|system|>")
            system_end = adjusted_prompt.find("<|user|>")
            
            if system_start != -1 and system_end != -1:
                system_message = adjusted_prompt[system_start:system_end]
                user_and_beyond = adjusted_prompt[system_end:]
                
                # Add detailed response instructions to system message
                if "Please provide detailed and comprehensive responses" not in system_message:
                    enhanced_system = system_message.rstrip() + " Please provide detailed and comprehensive responses with examples and explanations when appropriate.\n"
                    adjusted_prompt = enhanced_system + user_and_beyond
        
        inputs = tokenizer(adjusted_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            try:
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=actual_max_tokens,
                    do_sample=True,
                    temperature=actual_temp,
                    top_p=topp,
                    top_k=topk,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=3 if response_length == "long" else 0,  # Avoid repeating 3-grams in long responses
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            except RuntimeError as e:
                # Handle CUDA out of memory errors specifically
                if "CUDA out of memory" in str(e):
                    logger.error("CUDA out of memory during generation. Cleaning up and retrying with lower settings.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Retry with more conservative settings
                    outputs = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=min(256, actual_max_tokens // 2),  # Reduce max tokens
                        do_sample=False,                          # No sampling
                        num_beams=1,                              # No beam search
                        repetition_penalty=1.0,                   # No repetition penalty
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                else:
                    raise

        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response.strip()
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        if "CUDA out of memory" in str(e):
            return "I'm currently experiencing high load. Please try a shorter message or try again later."
        return "I encountered an error while processing your request. Please try again."

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests with improved session management and response length control"""
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
        response_length = data.get('response_length', 'medium')  # New parameter for response length
        
        # Validate response_length
        if response_length not in ['short', 'medium', 'long']:
            response_length = 'medium'  # Default to medium if invalid

        if not user_message:
            return jsonify({
                'status': 'error',
                'message': 'Empty message'
            }), 400

        # Input validation
        if len(user_message) > 4000:  # Reasonable limit
            return jsonify({
                'status': 'error',
                'message': 'Message too long. Please send a shorter message.'
            }), 400

        with session_lock:
            # Initialize session if needed
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
                
            # Check session length and trim if needed
            if len(chat_sessions[session_id]) >= MAX_SESSION_LENGTH * 2:  # *2 because each exchange has 2 messages
                # Keep only the most recent conversations
                chat_sessions[session_id] = chat_sessions[session_id][-(MAX_SESSION_LENGTH*2-2):]
                # Add a system message noting the truncation
                truncation_message = {
                    "role": "system",
                    "content": "[Some older messages have been removed to manage conversation length]",
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
                chat_sessions[session_id].insert(0, truncation_message)
                
            # Add user message to session
            chat_sessions[session_id].append({
                "role": "user",
                "content": user_message,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })

        # Generate response with specified length
        prompt = format_chat_history(session_id)
        response = generate_response(prompt, response_length=response_length)

        with session_lock:
            # Add assistant response to session
            chat_sessions[session_id].append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })

            # Update session timestamp
            session_last_active[session_id] = datetime.datetime.now()

        session_length = len(chat_sessions.get(session_id, []))
        return jsonify({
            'status': 'success',
            'response': response,
            'session_id': session_id,
            'session_messages': session_length,
            'session_limit': MAX_SESSION_LENGTH * 2,
            'response_length': response_length
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
        
        with session_lock:
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

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List active sessions with basic info"""
    try:
        with session_lock:
            sessions_info = {}
            for sid, messages in chat_sessions.items():
                last_active = session_last_active.get(sid)
                sessions_info[sid] = {
                    'message_count': len(messages),
                    'last_active': last_active.isoformat() if last_active else None,
                    'expires_in': f"{SESSION_TIMEOUT - (datetime.datetime.now() - last_active).total_seconds():.1f} seconds" if last_active else None
                }
        
        return jsonify({
            'status': 'success',
            'session_count': len(sessions_info),
            'sessions': sessions_info
        })
    except Exception as e:
        logger.error(f"Error in list_sessions endpoint: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/response_settings', methods=['GET'])
def get_response_settings():
    """Get available response settings"""
    return jsonify({
        'status': 'success',
        'max_tokens': MAX_TOKENS,
        'response_lengths': {
            'short': 'Brief responses (up to 256 tokens)',
            'medium': 'Standard balanced responses',
            'long': 'Detailed comprehensive responses (up to 1024 tokens)'
        },
        'current_model': MODEL_NAME
    })

@app.route('/')
def serve_index():
    """Serve the index.html file from the root directory"""
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}", exc_info=True)
        return "Error loading index.html. Make sure the file exists in the application directory.", 500

def is_wsl():
    """Improved WSL detection"""
    # Check for WSL-specific files
    if os.path.exists('/proc/sys/fs/binfmt_misc/WSLInterop'):
        return True
    
    # Check /proc/version content
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower() or "wsl" in f.read().lower()
    except FileNotFoundError:
        pass
    
    # Check environment variables
    return 'WSL_DISTRO_NAME' in os.environ

def open_browser(port):
    """Improved browser opening function with better platform detection"""
    url = f"http://localhost:{port}"
    
    time.sleep(2)  # Give the server a moment to start
    
    system = platform.system()
    
    try:
        # Handle WSL specifically
        if is_wsl():
            logger.info("WSL environment detected")
            try:
                # Try to use wslview which is the proper way to open URLs from WSL
                subprocess.run(["wslview", url], check=False)
                return
            except FileNotFoundError:
                # Fallback to PowerShell
                try:
                    subprocess.run(["powershell.exe", "-c", f"start '{url}'"], check=False)
                    return
                except Exception as e:
                    logger.warning(f"Could not open browser in WSL using PowerShell: {e}")
        
        # Handle Windows
        elif system == "Windows":
            try:
                os.startfile(url)
                return
            except AttributeError:
                # Fallback to PowerShell
                try:
                    subprocess.run(["powershell.exe", "-c", f"start '{url}'"], check=False)
                    return
                except Exception as e:
                    logger.warning(f"Could not open browser on Windows: {e}")
        
        # Handle macOS
        elif system == "Darwin":
            try:
                subprocess.run(["open", url], check=False)
                return
            except Exception as e:
                logger.warning(f"Could not open browser on macOS: {e}")
        
        # Handle Linux
        elif system == "Linux":
            try:
                subprocess.run(["xdg-open", url], check=False)
                return
            except Exception as e:
                logger.warning(f"Could not open browser on Linux: {e}")
        
        # Fallback to Python's webbrowser
        logger.info(f"Using Python webbrowser module to open {url}")
        webbrowser.open(url)
    
    except Exception as e:
        logger.warning(f"Failed to open browser: {e}")
        logger.info(f"Server is running at {url}. Please open this URL in your browser manually.")

def cleanup_sessions():
    """Improved session cleanup with better concurrency handling"""
    while True:
        try:
            now = datetime.datetime.now()
            
            with session_lock:
                # Find expired sessions
                expired_sessions = [
                    sid for sid, last_active in session_last_active.items()
                    if (now - last_active).total_seconds() > SESSION_TIMEOUT
                ]
                
                # Remove expired sessions
                for sid in expired_sessions:
                    chat_sessions.pop(sid, None)
                    session_last_active.pop(sid, None)
                    logger.info(f"Session {sid} cleaned up due to inactivity")
            
            # Memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Log memory usage periodically
            if model_loaded and torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                logger.info(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
            
            time.sleep(MEMORY_CLEANUP_INTERVAL)
        except Exception as e:
            logger.error(f"Error in cleanup_sessions: {e}", exc_info=True)
            time.sleep(60)  # Wait a minute before trying again

if __name__ == '__main__':
    # Check for required packages
    try:
        import psutil
    except ImportError:
        logger.warning("psutil package not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            import psutil
            logger.info("psutil installed successfully")
        except Exception as e:
            logger.error(f"Failed to install psutil: {e}")
    
    # Validate environment
    validate_environment()
    
    # Start model loading in a separate thread
    threading.Thread(target=setup_model, daemon=True).start()
    
    # Start session cleanup in a separate thread
    threading.Thread(target=cleanup_sessions, daemon=True).start()
    
    # Start Flask app and open web browser
    port = int(os.environ.get("PORT", 5000))
    logger = logging.getLogger(__name__)
    
    # Open browser in a separate thread
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()

    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on port {port}")
    # Disable reloading when in debug mode to avoid duplicate model loading
    app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)