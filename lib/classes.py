import asyncio
from abc import ABC, abstractmethod
from lib.configs import LLMConfig


class LLMService(ABC):
    """Abstract interface for all LLM services used in the experiment."""

    service_name: str
    provider: str

    @abstractmethod
    async def generate(self, prompt: str, config: LLMConfig) -> str:
        """Return the model response for a prompt."""
        raise NotImplementedError

class NonSpecificMockLLMService(LLMService):
    """A mock service for testing the pipeline without external API calls."""

    def __init__(self, service_name: str = "mock_service", provider: str = "local") -> None:
        self.service_name = service_name
        self.provider = provider

    async def generate(self, prompt: str, config: LLMConfig) -> str:
        await asyncio.sleep(0.05)
        return (
            "A realistic profile might describe an adult urban resident living in a rented flat, "
            "commuting by public transport, working in a service or office role, and balancing "
            "moderate financial pressure with access to nearby shops, clinics, and public spaces."
        )


class SpecificMockLLMService(LLMService):
    """A mock service for testing the pipeline without external API calls."""

    def __init__(self, service_name: str = "mock_service", provider: str = "local") -> None:
        self.service_name = service_name
        self.provider = provider

    async def generate(self, prompt: str, config: LLMConfig) -> str:
        await asyncio.sleep(0.05)
        return (
            "A realistic profile might describe Maya, a 34-year-old woman of South Asian ethnicity "
            "living with her partner and young child in a privately rented two-bedroom flat. She works "
            "full-time as an administrative coordinator, earns a lower-middle income, and is financially "
            "stable but has limited savings after rent, childcare, transport, and utility costs. She has "
            "no specific physical disability or chronic health condition mentioned, commutes by bus and "
            "underground train, and has regular access to local schools, GP clinics, supermarkets, public "
            "parks, and digital services."
        )

class ImpliedMockLLMService(LLMService):
    """A mock service for testing the pipeline without external API calls."""

    def __init__(self, service_name: str = "mock_service", provider: str = "local") -> None:
        self.service_name = service_name
        self.provider = provider

    async def generate(self, prompt: str, config: LLMConfig) -> str:
        await asyncio.sleep(0.05)
        return (
            "A realistic profile might describe Priya, a 34-year-old urban resident who lives with a "
            "partner and young child in a privately rented two-bedroom flat above a row of small shops. "
            "At home, the family speaks both English and a South Asian heritage language, and weekends "
            "often involve visiting relatives, shopping at specialist grocery stores, and attending local "
            "community events. Priya manages most school drop-offs and healthcare appointments around a "
            "full-time administrative job, commuting by bus and underground train. The household usually "
            "covers rent, childcare, transport, and utility bills on time, but there is little left over "
            "for savings or unexpected costs. Nearby services include a GP clinic, primary school, public "
            "parks, supermarkets, and digital council services."
        )

