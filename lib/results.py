from typing import Any, Dict
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RetrievalResult:
    run_id: str
    timestamp_utc: str
    collection_wave: str
    service_name: str
    service_provider: str
    service_model: str
    service_config_json: str
    prompt: str
    raw_response: str
    response_status: str
    error_message: str = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalysisResult:
    analysis_id: str
    run_id: str
    analysis_timestamp_utc: str
    analysis_service_name: str
    analysis_service_provider: str
    analysis_service_model: str
    analysis_service_config_json: str
    analysis_schema_version: str
    analysis_raw_json: str
    flattened_fields: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        base = {
            "analysis_id": self.analysis_id,
            "run_id": self.run_id,
            "analysis_timestamp_utc": self.analysis_timestamp_utc,
            "analysis_service_name": self.analysis_service_name,
            "analysis_service_provider": self.analysis_service_provider,
            "analysis_service_model": self.analysis_service_model,
            "analysis_service_config_json": self.analysis_service_config_json,
            "analysis_schema_version": self.analysis_schema_version,
            "analysis_raw_json": self.analysis_raw_json,
        }
        base.update(self.flattened_fields)
        return base
