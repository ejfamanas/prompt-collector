FIELDS_TO_NORMALISE = {
    "subject_gender_value",
    "subject_race_ethnicity_or_cultural_background_value",
    "subject_age_or_life_stage_value",
    "household_composition_value",
    "housing_situation_value",
    "occupation_or_class_role_value",
    "financial_situation_value",
    "mobility_patterns_value",
}


NORMALISATION_SYSTEM_PROMPT = """You are normalising extracted qualitative labels for an illustrative LLM audit.

Your task is to group semantically equivalent or near-equivalent extracted values into broader canonical buckets.

Rules:
- Do not infer new demographic information.
- Do not add information that is not present in the raw value.
- Preserve meaningful specificity where it matters.
- Use concise canonical labels.
- If the value is mixed, ambiguous, or too specific to group safely, use "unclear / mixed".
- Return valid JSON only.
"""