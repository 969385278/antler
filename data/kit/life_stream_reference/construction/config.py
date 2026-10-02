"""
Configuration file for Elias Zhang Story Generation Pipeline
Edit the API_KEY variable below with your Harvard API Portal key
"""

import os

# =============================================================================
# HARVARD OPENAI API CONFIGURATION
# =============================================================================
#
# How to get your API key:
# 1. Go to: https://portal.apis.huit.harvard.edu/
# 2. Register your app (you'll need a HUIT Billing ID)
# 3. After saving your app, you'll receive a key and secret
# 4. Copy the KEY (not the secret) and paste it below
#
# =============================================================================

# PASTE YOUR API KEY HERE:
API_KEY = os.environ.get("HARVARD_API_KEY", "YOUR_API_KEY_HERE")  # <-- Set via environment variable or edit here

# API Configuration
# API_BASE_URL = "https://go.apis.huit.harvard.edu/ais-google-gemini/v1beta"
# MODEL_NAME = "gemini-2.5-pro"  # Available: gemini-2.5-pro, gemini-2.0-flash

API_BASE_URL = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions"
# API_BASE_URL = "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions"
MODEL_NAME = "gpt-5.2-2025-12-11"

# Generation Configuration
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 100000

# =============================================================================
# DO NOT EDIT BELOW THIS LINE
# =============================================================================

def get_api_config():
    """Returns the API configuration."""
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n" + "="*80)
        print("⚠️  WARNING: API KEY NOT SET!")
        print("="*80)
        print("Please edit config.py and set your Harvard API Portal key.")
        print("Get your key from: https://portal.apis.huit.harvard.edu/")
        print("="*80 + "\n")
    
    return {
        "api_key": API_KEY,
        "base_url": API_BASE_URL,
        "model": MODEL_NAME,
        "url": API_BASE_URL,  # OpenAI endpoint (already complete URL)
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "max_output_tokens": MAX_OUTPUT_TOKENS
    }

