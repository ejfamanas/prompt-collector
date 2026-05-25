"""Hugging Face service implementation for the prompt collection pipeline."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from huggingface_hub import InferenceClient

from lib.classes import LLMService
from lib.configs import LLMConfig
from lib.schema import HUGGINGFACE_CHAT_ALLOWED_EXTRA_KEYS


def _apply_extra_config(request_payload: dict[str, Any], extra: dict[str, Any]) -> None:
    """Validate and apply Hugging Face-specific extra config values."""
    unsupported_keys = set(extra) - HUGGINGFACE_CHAT_ALLOWED_EXTRA_KEYS
    if unsupported_keys:
        unsupported = ", ".join(sorted(unsupported_keys))
        raise ValueError(
            f"Unsupported Hugging Face chat config key(s): {unsupported}. "
            "Only provider API parameters should be passed through LLMConfig.extra."
        )

    request_payload.update(extra)


class HuggingFaceChatService(LLMService):
    """Hugging Face implementation of the generic LLMService interface."""

    def __init__(
            self,
            service_name: str = "huggingface_chat",
            provider: str = "huggingface",
            api_key: str | None = None,
    ) -> None:
        self.service_name = service_name
        self.provider = provider
        self.client = InferenceClient(
            api_key=api_key or os.environ.get("HF_TOKEN")
        )

    async def generate(self, prompt: str, config: LLMConfig) -> str:
        """Generate text using Hugging Face chat completion."""
        request_payload: dict[str, Any] = {
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }

        if config.temperature is not None:
            request_payload["temperature"] = config.temperature

        if config.max_tokens is not None:
            request_payload["max_tokens"] = config.max_tokens

        if config.top_p is not None:
            request_payload["top_p"] = config.top_p

        if config.seed is not None:
            request_payload["seed"] = config.seed

        _apply_extra_config(request_payload, config.extra)

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            **request_payload,
        )

        content = response.choices[0].message.content
        return content or ""
