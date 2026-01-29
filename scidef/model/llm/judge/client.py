"""
LLM services: async-only client, prompt templates, and judge evaluation.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

from scidef.model.dataclass import JudgeSystemPrompt, ProviderType
from scidef.model.llm.client import LLMClient
from scidef.model.llm.judge.prompts import (
    get_judge_prompt,
    get_judge_system_prompt,
    parse_judge_response,
)
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


class JudgeClient(LLMClient):
    """LLM-based definition comparison and judgment."""

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

    async def judge_definition_pair(
        self,
        human_definition: str,
        model_definition: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt_type: Optional[JudgeSystemPrompt] = None,
    ) -> Tuple[Optional[Union[int, float]], Optional[str], Optional[str]]:
        """Judge the similarity between human and model definitions."""
        try:
            prompt = get_judge_prompt(
                human_definition,
                model_definition,
            )
            system_prompt = get_judge_system_prompt(system_prompt_type)

            response, _ = await super().generate_text(
                prompt,
                temperature=temperature,
                top_p=top_p,
                system_prompt=system_prompt,
            )
            judgment_category = parse_judge_response(response)
            return judgment_category, response, None

        except Exception as e:
            error_msg = f"Error in LLM judge evaluation: {e}"
            logger.error(error_msg)
            return None, None, error_msg
