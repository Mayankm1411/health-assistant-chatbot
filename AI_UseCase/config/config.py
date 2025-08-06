
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
