import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm():
    """Returns Gemini 1.5 Flash LLM instance."""
    api_key = os.getenv("Gemini_API_KEY")
    if not api_key:
        raise EnvironmentError("Gemini_API_KEY not set in environment variables.")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )
