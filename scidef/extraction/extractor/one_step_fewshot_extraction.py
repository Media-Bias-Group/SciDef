from typing import Any, Dict, List, Optional

from scidef.extraction.extractor.base import BaseExtractor


def indentify_extract_llm(text):
    SYSTEM = """You are an expert in definition extraction for academic papers. Extract definitions directly from the given text.

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
    If MULTIPLE definitions are present, return a JSON array of objects.
    Quotes escaped by single quotes `'`

    Return only valid JSON matching this schema:
    {
      "term": "string",
      "type": "explicit" | "implicit",
      "definition": "string"
    }

    Examples:

    Input: "No formal definition exists but there is a consensus that it is speech that targets disadvantaged social groups in a manner that is potentially harmful to them."
    Output (JSON!):
    {
    "term": "hate speech",
    "type": "explicit",
    "definition": "speech that targets disadvantaged social groups in a manner that is potentially harmful to them"
    }

    Input: "If firms are aware of the advantages of forming prior alliances, then our findings could exhibit endogeneity driven by selection bias-that is, superior firms may intentionally select to engage in prior alliances at a higher rate than inferior firms. Thus, the OLS coefficient estimates of Prior Alliance could be biased upward due to self-selection."
    Output (JSON!):
    {
    "term": "selection bias",
    "type": "implicit",
    "definition": "superior firms may intentionally select to engage in prior alliances at a higher rate than inferior firms"
    }"""

    INSTRUCTION = f"""Extract all definitions from this text:
    ```
    {text}
    ```
    """

    return SYSTEM, INSTRUCTION, {"term", "type", "definition"}


class OnesStepFewShotExtractor(BaseExtractor):
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
        (
            extract_system,
            extract_instruction,
            required_keys,
        ) = indentify_extract_llm(text)

        extraction_response, _ = await self.llm_client.generate_text(
            prompt=extract_instruction,
            system_prompt=extract_system,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return self._safe_parse_output(
            input_str=text,
            response_str=extraction_response,
            required_keys=required_keys,
        )
