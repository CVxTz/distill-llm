import os
from pathlib import Path

BASE_PATH = Path(__file__).parents[1]


TEMPERATURE = 0.0

BASE_URL = "https://api.endpoints.anyscale.com/v1"
# BASE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
BASE_MODEL = "meta-llama/Llama-2-70b-chat-hf"

API_KEY = os.getenv("API_KEY")
