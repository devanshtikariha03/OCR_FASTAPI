import os
from dotenv import load_dotenv

def load_env():
    # loads .env if present; in prod use actual secret manager
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return api_key, mode