import uuid
from datetime import datetime, timezone
from pathlib import Path
from lib.schema import RETRIEVAL_COLUMNS
from lib.results import RetrievalResult
from lib.classes import LLMService
from lib.configs import LLMConfig
from utils.csv_utils import append_csv_row


async def retrieve_response(
        *,
        prompt: str,
        service: LLMService,
        config: LLMConfig,
        retrieval_csv_path: Path,
        collection_wave: str = "T0",
) -> RetrievalResult:
    """Call a specified LLM service asynchronously and store the response.

    Parameters
    ----------
    prompt:
        The exact experiment prompt.
    service:
        The LLM service implementation to call.
    config:
        Generic config object associated with the service call.
    retrieval_csv_path:
        CSV path where the retrieval row will be appended.
    collection_wave:
        Longitudinal wave label, such as T0, T1, T2.

    Returns
    -------
    RetrievalResult
        The structured retrieval metadata and raw model response.
    """

    run_id = str(uuid.uuid4())
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    try:
        raw_response = await service.generate(prompt, config)
        result = RetrievalResult(
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            collection_wave=collection_wave,
            service_name=service.service_name,
            service_provider=service.provider,
            service_model=config.model,
            service_config_json=config.to_json(),
            prompt=prompt,
            raw_response=raw_response,
            response_status="success",
            error_message="",
        )
    except Exception as exc:  # noqa: BLE001 - preserve experiment failure details
        result = RetrievalResult(
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            collection_wave=collection_wave,
            service_name=service.service_name,
            service_provider=service.provider,
            service_model=config.model,
            service_config_json=config.to_json(),
            prompt=prompt,
            raw_response="",
            response_status="error",
            error_message=repr(exc),
        )

    append_csv_row(retrieval_csv_path, RETRIEVAL_COLUMNS, result.to_row())
    return result
