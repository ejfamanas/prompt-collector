from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from lib.configs import LLMConfig
from services.openai_service import OpenAIResponsesService
from utils.csv_utils import read_csv_rows, write_csv_rows
from utils.json_utils import parse_json_object


SHARED_THEME_OUTPUT_COLUMNS = [
    "theme_field",
    "theme_value",
    "normalised_theme_value",
    "model_count",
    "models",
    "total_count_across_models",
    "canonical_theme_value",
    "normalisation_confidence",
    "normalisation_rationale",
]


SHARED_THEME_NORMALISATION_PROMPT = """You are normalising shared theme values from an illustrative LLM audit.

Your task is to collapse near-equivalent shared theme values into broader canonical buckets for graphing.

Rules:
- Do not infer new demographic information.
- Do not add information that is not present in the raw values.
- Group semantically equivalent or near-equivalent values.
- Preserve meaningful specificity where it matters.
- Use concise canonical labels.
- If a value is mixed, ambiguous, or too specific to group safely, use "unclear / mixed".
- Return valid JSON only.
"""


def build_shared_theme_normalisation_prompt(theme_field: str, theme_values: list[str]) -> str:
    """Build one prompt for all shared values belonging to the same theme field."""
    values_as_json = json.dumps(theme_values, ensure_ascii=False, indent=2)
    return f"""
Normalise these shared theme values for one theme field.

Theme field:
{theme_field}

Raw shared theme values:
{values_as_json}

Return JSON with this exact shape:
{{
  "mappings": [
    {{
      "theme_value": "exact original value from the input list",
      "canonical_theme_value": "broader graphing bucket",
      "confidence": "0.0 to 1.0 as a string",
      "rationale": "brief explanation"
    }}
  ]
}}
""".strip()


def group_rows_by_theme_field(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group shared-theme rows by theme field."""
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        theme_field = row.get("theme_field", "").strip()
        if theme_field:
            grouped_rows[theme_field].append(row)
    return dict(grouped_rows)


def parse_mapping_response(raw_response: str) -> dict[str, dict[str, str]]:
    """Parse the LLM mapping response into a lookup keyed by original theme_value."""
    parsed = parse_json_object(raw_response)
    mappings = parsed.get("mappings", [])
    if not isinstance(mappings, list):
        raise ValueError("Normalisation response did not contain a mappings list.")

    mapping_by_theme_value: dict[str, dict[str, str]] = {}
    for mapping in mappings:
        if not isinstance(mapping, dict):
            continue

        theme_value = str(mapping.get("theme_value", "")).strip()
        if not theme_value:
            continue

        mapping_by_theme_value[theme_value] = {
            "canonical_theme_value": str(mapping.get("canonical_theme_value", "unclear / mixed")),
            "normalisation_confidence": str(mapping.get("confidence", "0.0")),
            "normalisation_rationale": str(mapping.get("rationale", "")),
        }

    return mapping_by_theme_value


async def normalise_shared_theme_field(
    *,
    analysis_service: OpenAIResponsesService,
    analysis_config: LLMConfig,
    theme_field: str,
    theme_values: list[str],
) -> dict[str, dict[str, str]]:
    """Normalise all shared values for one theme field using one LLM call."""
    prompt = SHARED_THEME_NORMALISATION_PROMPT + "\n\n" + build_shared_theme_normalisation_prompt(
        theme_field=theme_field,
        theme_values=theme_values,
    )

    raw_response = await analysis_service.generate(prompt, analysis_config)
    return parse_mapping_response(raw_response)


async def normalise_shared_theme_summary(
    input_shared_theme_summary_path: Path,
    output_normalised_shared_theme_summary_path: Path,
) -> None:
    """Create a graph-ready normalised shared-theme summary CSV."""
    load_dotenv()

    rows = read_csv_rows(input_shared_theme_summary_path)
    grouped_rows = group_rows_by_theme_field(rows)

    analysis_service = OpenAIResponsesService(
        service_name="openai_shared_theme_normaliser",
        provider="openai",
    )
    analysis_config = LLMConfig(
        model="gpt-5.2",
        temperature=0.0,
        max_tokens=3000,
        extra={
            "store": False,
            "text": {"format": {"type": "json_object"}},
        },
    )

    write_csv_rows(output_normalised_shared_theme_summary_path, SHARED_THEME_OUTPUT_COLUMNS, [], mode="overwrite")

    for index, (theme_field, field_rows) in enumerate(grouped_rows.items(), start=1):
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"[{timestamp}] Processing theme field {index}/{len(grouped_rows)}: {theme_field}")

        theme_values = [
            row.get("theme_value", "").strip()
            for row in field_rows
            if row.get("theme_value", "").strip()
        ]

        try:
            mappings = await normalise_shared_theme_field(
                analysis_service=analysis_service,
                analysis_config=analysis_config,
                theme_field=theme_field,
                theme_values=theme_values,
            )
        except Exception as error:
            mappings = {
                theme_value: {
                    "canonical_theme_value": "normalisation_error",
                    "normalisation_confidence": "0.0",
                    "normalisation_rationale": repr(error),
                }
                for theme_value in theme_values
            }

        for row in field_rows:
            theme_value = row.get("theme_value", "").strip()
            normalised = mappings.get(
                theme_value,
                {
                    "canonical_theme_value": "unclear / mixed",
                    "normalisation_confidence": "0.0",
                    "normalisation_rationale": "No mapping returned for this shared theme value.",
                },
            )

            output_row = {
                "theme_field": row.get("theme_field", ""),
                "theme_value": row.get("theme_value", ""),
                "normalised_theme_value": row.get("normalised_theme_value", ""),
                "model_count": row.get("model_count", ""),
                "models": row.get("models", ""),
                "total_count_across_models": row.get("total_count_across_models", ""),
                **normalised,
            }
            write_csv_rows(output_normalised_shared_theme_summary_path, SHARED_THEME_OUTPUT_COLUMNS, [output_row], mode="append")

        await asyncio.sleep(0.5)

    print(f"Wrote normalised shared summary to {output_normalised_shared_theme_summary_path}")