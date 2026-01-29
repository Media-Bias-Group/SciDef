import ast
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from scidef.model.llm.client import LLMClient
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


class BaseExtractor(ABC):
    def __init__(
        self,
        llm_client: LLMClient,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> None:
        self.llm_client = llm_client
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    async def run(self, text: str) -> List[Dict[Any, Any]]:
        raise NotImplementedError("not Implemented!")

    def _safe_parse_output(
        self,
        input_str: str,
        response_str: str,
        required_keys: Set[str] = {"term", "type", "definition"},
    ) -> List[Dict[str, str]]:
        try:
            # remove thinking
            if "think" in response_str.lower():
                response_str = response_str.split("think>")[-1].strip()

            # Remove markdown code blocks markers
            response_str = re.sub(r"```\w*", "", response_str)
            response_str = response_str.replace("```", "")

            # remove everything apart from `{...}`
            start_idx = response_str.find("{")
            end_idx = response_str.rfind("}")

            if start_idx != -1 and end_idx != -1:
                response_str = response_str[start_idx : end_idx + 1]
            else:
                logger.warning(
                    f"Response doesn't fit the json format: {response_str}",
                )
                return []

            # deal with multiple json objects in the response
            if response_str.startswith("{") and response_str.count("{") > 1:
                response_str = (
                    response_str.replace("\n", "")
                    .replace(
                        "}{",
                        "},{",
                    )
                    .replace("} {", "},{")
                )
                response_str = f"[{response_str}]"

            try:
                data = json.loads(response_str)
            except json.JSONDecodeError as je:
                try:
                    data = ast.literal_eval(response_str)
                except Exception as ee:
                    logger.error(
                        f"AST & JSON decoding error for response: {response_str}\n JSON Error: {je}\n AST Error: {ee}",
                    )
                    return []

            if isinstance(data, dict):
                if not data:
                    return []
                if required_keys.issubset(data.keys()):
                    data["input"] = input_str
                    return [data]
                else:
                    logger.warning(
                        f"Response doesn't fit the correct format (after json was loaded): {response_str}",
                    )
                    return []

            elif isinstance(data, list):
                valid_items = [
                    item
                    for item in data
                    if isinstance(item, dict)
                    and required_keys.issubset(item.keys())
                ]
                for item in valid_items:
                    item["input"] = input_str
                return valid_items if valid_items else []

            else:
                logger.warning(
                    f"Response doesn't fit the correct format (after json was loaded): {response_str}",
                )
                return []
        except Exception as e:
            logger.warning(
                f"Error parsing the LLM response: {e} ({response_str})",
            )
            return []

    def _safe_parse_binary_output(
        self,
        response_str: str,
    ) -> Optional[bool]:
        try:
            response_clean = response_str.strip().lower()

            # remove thinking
            if "think" in response_str.lower():
                response_str = response_str.split("think>")[-1].strip()

            if ("yes" in response_clean) or ("no" in response_clean):
                return "yes" in response_clean
            else:
                logger.warning("Binary response not recognized: %s", response_str)
                return None
        except Exception:
            return False
