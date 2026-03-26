import os
import logging
from dotenv import load_dotenv, find_dotenv

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

def load_env():
    """Load environment variables from .env file."""
    _ = load_dotenv(find_dotenv())

def suppress_warnings():
    """Suppress specific library warnings."""
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

def get_openai_api_key():
    """Get OpenAI API key from environment."""
    load_env()
    return os.getenv("OPENAI_API_KEY")

# Model token limits (for context management)
MODEL_TOKEN_LIMITS = {
    "gpt-5": 256000,
    "gpt-5-mini": 256000,
}
