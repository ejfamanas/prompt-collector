import json
from typing import Any


def parse_json_object(raw_response: str) -> dict[str, Any]:
    """Parse a JSON object returned by the normalisation model."""
    parsed = json.loads(raw_response)
    if not isinstance(parsed, dict):
        raise ValueError("Normalisation response was not a JSON object.")
    return parsed