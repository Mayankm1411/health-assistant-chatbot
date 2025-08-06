import os
from langchain_groq import ChatGroq
from config.config import GROQ_API_KEY  # Make sure this is correctly set

def get_chat_model():
    return ChatGroq(
        temperature=0.3,
        model_name="llama3-70b-8192",  # Or "llama3-8b-8192" for a smaller one
        api_key=GROQ_API_KEY
    )
