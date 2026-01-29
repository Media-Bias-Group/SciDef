"""
LLM services: async client for definition extraction.
"""

from pathlib import Path
from typing import Optional

from scidef.model.dataclass import ProviderType
from scidef.model.llm.client import LLMClient
from scidef.model.llm.extraction.prompts import (
    ExtractionPrompt,
    get_extraction_prompt,
)
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


class ExtractionClient(LLMClient):
    """Async-only OpenAI-compatible LLM client for definition extraction."""

    def __init__(
        self,
        provider_type: ProviderType,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__(
            provider_type,
            base_url,
            api_key,
            model_name,
            timeout,
            max_retries,
            cache_dir,
        )

    async def extract_definitions(
        self,
        text: str,
        prompt_type: ExtractionPrompt,
    ) -> tuple[str, float]:
        """Extract definitions using the language model asynchronously."""
        prompt = get_extraction_prompt(prompt_type, text)
        return await super().generate_text(
            prompt if prompt else "",
        )
