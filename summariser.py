"""Post-collection theme normalisation for graph-ready summaries.

This script is intentionally separate from main.py. It reads the exact-match
model theme summary produced by the collection pipeline, asks an analysis LLM to
collapse semantically similar values into canonical buckets, and writes a new
CSV suitable for visualisation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from utils.normalisers import normalise_theme_summary


INPUT_MODEL_THEME_SUMMARY_PATH = Path("data/model_theme_summary.csv")
OUTPUT_NORMALISED_THEME_SUMMARY_PATH = Path("data/normalised_model_theme_summary.csv")


if __name__ == "__main__":
    asyncio.run(
        normalise_theme_summary(
            INPUT_MODEL_THEME_SUMMARY_PATH,OUTPUT_NORMALISED_THEME_SUMMARY_PATH
        ))
