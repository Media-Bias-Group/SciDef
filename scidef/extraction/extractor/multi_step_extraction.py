from typing import Any, Dict, List, Optional

from scidef.extraction.extractor.base import BaseExtractor
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


def create_binary_llm(text):
    SYSTEM = """You are an expert in definition detection for academic papers. Determine whether the text contains a definition of a term or concept.

A definition explains what a concept IS, not just mentions it or provides examples.

DEFINITIONS:
- "X is/means/refers to/denotes..."
- "We define X as..."
- Explanations using colons, apposition, or characteristics
- Conceptual discussions explaining meaning

NOT DEFINITIONS:
- Mere mentions or lists of examples
- Method descriptions or citations to other sources
- Citation-only references (see Smith, 2021)

Respond with only "YES" or "NO"."""

    INSTRUCTION = f"Does this contain a definition?\n```\n{text}\n```\nAnswer:"

    return SYSTEM, INSTRUCTION


def create_extraction_llm(text):
    SYSTEM = """You are an expert in definition extraction for academic papers. Extract a definition from the given section.

For each definition, identify:
1. term: The concept being defined
2. type: "explicit" or "implicit"
   - Explicit: Uses clear definitional patterns ("X is...", "X refers to...", "We define X as...", "X denotes...", "By X we mean...")
   - Implicit: Explains through characteristics, functions, boundaries, lists, colons, apposition, or conceptual discussion without explicit verbs
3. definition: The cleaned definition text (see extraction rules below)

EXTRACTION RULES:
1. Remove leading phrases: Delete "X is defined as", "X refers to", "X means"
2. Remove citations from definition: Delete "(Author, Year)" or "[14]" from the definition text itself
3. Extract minimal span: Include only the core definitional content

If NO definition is present, return an empty object: {}
If ONE definition is present, return a single JSON object.
If MULTIPLE definitions are present, return a list of JSON objects enclosed in [].
Quotes need to be escaped by single quotes `'`

Return only valid JSON matching this schema:
{
  "term": "string",
  "type": "explicit" | "implicit",
  "definition": "string"
}
"""

    INSTRUCTION = f"""Extract all definitions from this text:
```
{text}
```
"""

    return SYSTEM, INSTRUCTION, {"term", "type", "definition"}  # lowercase


class MultiStepExtractor(BaseExtractor):
    def __init__(
        self,
        llm_client,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__(
            llm_client=llm_client,
            temperature=temperature,
            top_p=top_p,
        )

    async def run(self, text: str) -> List[Dict[Any, Any]]:
        binary_system, binary_instruction = create_binary_llm(text)
        (
            extract_system,
            extract_instruction,
            required_keys,
        ) = create_extraction_llm(text)

        binary_response, _ = await self.llm_client.generate_text(
            prompt=binary_instruction,
            system_prompt=binary_system,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        binary_response = self._safe_parse_binary_output(binary_response)
        if binary_response is None:
            logger.warning(
                f"!!!!!!!!!!!{binary_system}\n{binary_instruction}\n{binary_response}!!!!!!!!!!!\n\n\n\n\n\n",
            )

        if binary_response:
            extracted_response, _ = await self.llm_client.generate_text(
                prompt=extract_instruction,
                system_prompt=extract_system,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            return self._safe_parse_output(
                input_str=text,
                response_str=extracted_response,
                required_keys=required_keys,
            )
        else:
            return []
