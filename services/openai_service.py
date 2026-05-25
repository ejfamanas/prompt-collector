"""OpenAI service implementation for the prompt collection pipeline."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import OpenAI

from lib.classes import LLMService
from lib.configs import LLMConfig
from lib.schema import OPENAI_RESPONSES_ALLOWED_EXTRA_KEYS


def _apply_extra_config(request_payload: dict[str, Any], extra: dict[str, Any]) -> None:
    """Validate and apply OpenAI-specific extra config values."""
    unsupported_keys = set(extra) - OPENAI_RESPONSES_ALLOWED_EXTRA_KEYS
    if unsupported_keys:
        unsupported = ", ".join(sorted(unsupported_keys))
        raise ValueError(
            f"Unsupported OpenAI Responses config key(s): {unsupported}. "
            "Only provider API parameters should be passed through LLMConfig.extra."
        )

    request_payload.update(extra)


class OpenAIResponsesService(LLMService):
    """OpenAI implementation of the generic LLMService interface."""

    def __init__(
        self,
        service_name: str = "openai_responses",
        provider: str = "openai",
        api_key: str | None = None,
    ) -> None:
        self.service_name = service_name
        self.provider = provider
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    async def generate(self, prompt: str, config: LLMConfig) -> str:
        """Generate text from OpenAI using the Responses API."""
        request_payload: dict[str, Any] = {
            "model": config.model,
            "input": prompt,
        }

        if config.temperature is not None:
            request_payload["temperature"] = config.temperature

        if config.max_tokens is not None:
            request_payload["max_output_tokens"] = config.max_tokens

        if config.top_p is not None:
            request_payload["top_p"] = config.top_p

        _apply_extra_config(request_payload, config.extra)

        response = await asyncio.to_thread(
            self.client.responses.create,
            **request_payload,
        )

        return response.output_text
