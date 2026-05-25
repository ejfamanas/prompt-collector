from typing import FrozenSet, List


RETRIEVAL_COLUMNS: List[str] = [
    "run_id",
    "timestamp_utc",
    "collection_wave",
    "service_name",
    "service_provider",
    "service_model",
    "service_config_json",
    "prompt",
    "raw_response",
    "response_status",
    "error_message",
]

# These are the locked analysis fields for the CUP illustrative experiment.
# Each field receives a structured code and supporting evidence.
ANALYSIS_FIELDS: List[str] = [
    "subject_gender",
    "subject_race_ethnicity_or_cultural_background",
    "subject_age_or_life_stage",
    "physical_features",
    "health_related_characteristics",
    "household_composition",
    "housing_situation",
    "occupation_or_class_role",
    "financial_situation",
    "mobility_patterns",
    "access_to_services",
    "everyday_environment",
    "cultural_or_geographic_specificity",
    "diversity_or_anti_generalisation_caveat",
    "refusal_or_safety_caveat",
    "stereotype_risk",
]

ANALYSIS_COLUMNS: List[str] = [
    "analysis_id",
    "run_id",
    "analysis_timestamp_utc",
    "analysis_service_name",
    "analysis_service_provider",
    "analysis_service_model",
    "analysis_service_config_json",
    "analysis_schema_version",
    "analysis_raw_json",
]

# Optional flattened columns for easier spreadsheet inspection.
# For each field we store: code, value, evidence, confidence.
FLATTENED_ANALYSIS_COLUMNS: List[str] = []
for field_name in ANALYSIS_FIELDS:
    FLATTENED_ANALYSIS_COLUMNS.extend(
        [
            f"{field_name}_code",
            f"{field_name}_value",
            f"{field_name}_evidence",
            f"{field_name}_confidence",
        ]
    )

FULL_ANALYSIS_COLUMNS: List[str] = ANALYSIS_COLUMNS + FLATTENED_ANALYSIS_COLUMNS

ANALYSIS_SCHEMA_VERSION = "representational_completion_v1"

VALID_FIELD_CODES = [
    "explicit",
    "implied",
    "unspecified",
    "problematised",
    "not_applicable",
]

VALID_BINARY_CODES = [
    "present",
    "absent",
    "partial",
    "not_applicable",
]

VALID_STEREOTYPE_RISK_CODES = [
    "low",
    "medium",
    "high",
    "unclear",
]

# Provider-specific API parameter allowlists.
# These define which keys are allowed inside LLMConfig.extra for each provider.
# They are intentionally conservative to avoid accidentally passing experiment
# metadata or unsupported parameters into provider API calls.
OPENAI_RESPONSES_ALLOWED_EXTRA_KEYS: FrozenSet[str] = frozenset(
    {
        "store",
        "text",
        "metadata",
        "reasoning",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "previous_response_id",
        "conversation",
        "instructions",
        "truncation",
        "user",
    }
)

HUGGINGFACE_CHAT_ALLOWED_EXTRA_KEYS: FrozenSet[str] = frozenset(
    {
        "provider",
        "stream",
        "stop",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "do_sample",
        "return_full_text",
    }
)
