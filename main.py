import os
from dotenv import load_dotenv
from prompt import get_prompt_template
from model import get_llm
from parser import parse_llm_output, EventInfo

load_dotenv()

def extract_event_info(event_text: str) -> EventInfo:
    """
    Full pipeline: takes raw event text, returns structured EventInfo.
    Pipeline:
      User Input → Prompt Template → LLM → Output Parser → Structured Output
    """
    prompt = get_prompt_template()
    llm = get_llm()  # No provider argument — Gemini only
    chain = prompt | llm
    response = chain.invoke({"event_text": event_text})
    raw_text = response.content if hasattr(response, "content") else str(response)
    return parse_llm_output(raw_text)


def run_cli():
    """CLI interface for the Event Information Extractor."""
    print("\n" + "=" * 50)
    print("   EVENT INFORMATION EXTRACTOR")
    print("   Powered by Generative AI")
    print("=" * 50)
    print("Type 'quit' to exit.\n")

    while True:
        print("Paste your event description below:")
        user_input = input("> ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not user_input:
            print("Please enter some text.\n")
            continue

        try:
            print("\nExtracting event info...")
            event = extract_event_info(user_input)
            print(event.display())
            print()
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    run_cli()