
from __future__ import annotations

import asyncio
from pathlib import Path

from lib.normaliser import FIELDS_TO_NORMALISE, NORMALISATION_SYSTEM_PROMPT
from utils.csv_utils import read_csv_rows, write_csv_rows
from utils.json_utils import parse_json_object

from lib.configs import LLMConfig
from services.openai_service import OpenAIResponsesService


def build_normalisation_prompt(theme_field: str, theme_value: str) -> str:
    """Build the LLM prompt for one theme value."""
    return f"""
Normalise the following extracted theme value.

Theme field:
{theme_field}

Raw extracted value:
{theme_value}

Return JSON with this exact shape:
{{
  "canonical_value": "...",
  "confidence": "0.0 to 1.0 as a string",
  "rationale": "brief explanation"
}}
""".strip()


async def normalise_one_value(
        *,
        analysis_service: OpenAIResponsesService,
        analysis_config: LLMConfig,
        theme_field: str,
        theme_value: str,
) -> dict[str, str]:
    """Normalise one raw theme value using the analysis LLM."""
    prompt = NORMALISATION_SYSTEM_PROMPT + "\n\n" + build_normalisation_prompt(theme_field, theme_value)

    raw_response = await analysis_service.generate(prompt, analysis_config)
    parsed = parse_json_object(raw_response)

    return {
        "canonical_value": str(parsed.get("canonical_value", "unclear / mixed")),
        "normalisation_confidence": str(parsed.get("confidence", "0.0")),
        "normalisation_rationale": str(parsed.get("rationale", "")),
    }


async def normalise_theme_summary(
        input_model_theme_summary_path: Path,
        output_normalized_theme_summary_path: Path,
) -> None:
    """Create a graph-ready normalised theme summary CSV."""
    load_dotenv()

    rows = read_csv_rows(input_model_theme_summary_path)
    analysis_service = OpenAIResponsesService(
        service_name="openai_theme_normaliser",
        provider="openai",
    )
    analysis_config = LLMConfig(
        model="gpt-5.2",
        temperature=0.0,
        max_tokens=800,
        extra={
            "store": False,
            "text": {"format": {"type": "json_object"}},
        },
    )

    cache: dict[tuple[str, str], dict[str, str]] = {}
    output_rows: list[dict[str, str]] = []

    for row in rows:
        theme_field = row.get("theme_field", "")
        theme_value = row.get("theme_value", "")

        if theme_field not in FIELDS_TO_NORMALISE or not theme_value.strip():
            normalised = {
                "canonical_value": theme_value,
                "normalisation_confidence": "1.0",
                "normalisation_rationale": "Field not selected for LLM-assisted normalisation.",
            }
        else:
            cache_key = (theme_field, theme_value.strip().lower())
            if cache_key not in cache:
                try:
                    cache[cache_key] = await normalise_one_value(
                        analysis_service=analysis_service,
                        analysis_config=analysis_config,
                        theme_field=theme_field,
                        theme_value=theme_value,
                    )
                except Exception as error:
                    cache[cache_key] = {
                        "canonical_value": "normalisation_error",
                        "normalisation_confidence": "0.0",
                        "normalisation_rationale": repr(error),
                    }
            normalised = cache[cache_key]

        output_rows.append(
            {
                **row,
                **normalised,
            }
        )

        await asyncio.sleep(0.5)

    if output_rows:
        fieldnames = list(output_rows[0].keys())
    else:
        fieldnames = [
            "service_provider",
            "service_model",
            "theme_field",
            "theme_value",
            "normalised_theme_value",
            "count",
            "total_model_runs",
            "percentage",
            "canonical_value",
            "normalisation_confidence",
            "normalisation_rationale",
        ]

    write_csv_rows(output_normalized_theme_summary_path, fieldnames, output_rows)
    print(f"Wrote normalised summary to {output_normalized_theme_summary_path}")