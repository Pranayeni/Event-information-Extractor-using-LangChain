from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """You are an expert event information extractor. 
Your job is to extract structured event details from unstructured text.

Rules:
- Extract ONLY information explicitly mentioned in the text
- Do NOT invent or guess any missing details
- If a field is not mentioned, return exactly: "not_available"
- Return ONLY valid JSON, no extra text or explanation

Output must follow this exact JSON schema:
{{
  "event_name": "string or not_available",
  "event_date": "string or not_available",
  "event_time": "string or not_available",
  "event_location": "string or not_available",
  "organizer": "string or not_available"
}}"""

HUMAN_PROMPT = """Extract event information from the following text:

{event_text}

Return only the JSON object."""

def get_prompt_template() -> ChatPromptTemplate:
    """Returns the ChatPromptTemplate for event extraction."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT)
    ])
