import json
from pydantic import BaseModel, Field
from typing import Optional

NOT_AVAILABLE = "not_available"

class EventInfo(BaseModel):
    """Pydantic model representing structured event information."""
    event_name: str = Field(default=NOT_AVAILABLE, description="Name of the event")
    event_date: str = Field(default=NOT_AVAILABLE, description="Date of the event")
    event_time: str = Field(default=NOT_AVAILABLE, description="Time of the event")
    event_location: str = Field(default=NOT_AVAILABLE, description="Location/venue of the event")
    organizer: str = Field(default=NOT_AVAILABLE, description="Organizer of the event")

    def to_dict(self) -> dict:
        return self.model_dump()

    def display(self) -> str:
        """Returns a human-readable string of the event info."""
        lines = [
            "=" * 40,
            "  EXTRACTED EVENT INFORMATION",
            "=" * 40,
            f"  Event Name  : {self.event_name}",
            f"  Date        : {self.event_date}",
            f"  Time        : {self.event_time}",
            f"  Location    : {self.event_location}",
            f"  Organizer   : {self.organizer}",
            "=" * 40,
        ]
        return "\n".join(lines)


def parse_llm_output(raw_output: str) -> EventInfo:
    """
    Parses the raw LLM string output into an EventInfo Pydantic model.
    Handles JSON fences and whitespace.
    """
    text = raw_output.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        data = json.loads(text)
        # Replace empty strings or None with not_available
        for field in EventInfo.model_fields:
            if not data.get(field):
                data[field] = "not_available"
        return EventInfo(**data)
    except (json.JSONDecodeError, Exception) as e:
        raise ValueError(f"Failed to parse LLM output as JSON: {e}\nRaw output:\n{raw_output}")
