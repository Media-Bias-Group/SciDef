"""
Extraction prompts for definition extraction from academic papers.
"""

import json
import re
from typing import List, Optional, Tuple

from scidef.model.dataclass import Definition, ExtractionPrompt


def get_extraction_prompt(
    prompt_type: ExtractionPrompt,
    article_text: str,
) -> Optional[str]:
    """Get formatted extraction prompt for given type and article text."""

    if prompt_type == ExtractionPrompt.EXTRACTIVE:
        return f"""Please read the following academic paper and extract all concept definitions you can find.
Look for both explicit definitions (using phrases like "is defined as", "refers to", "means")
and implicit definitions where concepts are explained or characterized.

For each definition you find, format it EXACTLY as:
Concept Name - Definition text

For example:
machine learning - A method of data analysis that automates analytical model building
neural network - A computing system inspired by biological neural networks

Here is the paper:

{article_text}

Please extract all concept definitions you can identify, using the exact format "Concept Name - Definition text":"""

    elif prompt_type == ExtractionPrompt.STRUCTURED:
        return f"""Please read the following academic paper and extract concept definitions in a structured format.
For each definition, provide it in this EXACT format:

Concept Name - Definition text
Type: [Explicit/Implicit/Inferred]
Section: [Abstract/Introduction/Methods/Results/Discussion/Conclusion]
Related: [Any related concepts mentioned, if any]

For example:
machine learning - A method of data analysis that automates analytical model building
Type: Explicit
Section: Introduction
Related: artificial intelligence, data science

Here is the paper:

{article_text}

Please extract definitions in the exact structured format shown above:"""

    elif prompt_type == ExtractionPrompt.JSON:
        return f"""Please read the following academic paper and extract all concept definitions you can find.
Look for both explicit definitions (using phrases like "is defined as", "refers to", "means")
and implicit definitions where concepts are explained or characterized.

You must respond with ONLY a valid JSON object in this exact format (no additional text):

{{
  "concept name 1": "definition text 1",
  "concept name 2": "definition text 2",
  "concept name 3": "definition text 3"
}}

Here is the paper:

{article_text}

Respond with ONLY the JSON object containing the extracted definitions:"""

    # should not happen if type checking is done correctly
    return None


def parse_extraction_response(
    response: str,
    prompt_type: ExtractionPrompt,
) -> Tuple[List[Definition], Optional[str]]:
    """Parse the LLM extraction response and return (definitions, thought_process).

    - Extracts optional <think>...</think> content as thought_process.
    - Removes the <think> block from the text before parsing definitions.
    """
    if not response:
        return [], None

    # Capture thought process if present, then remove it
    thought_match = re.search(
        r"<think>(.*?)</think>",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    thought_process: Optional[str] = (
        thought_match.group(1).strip() if thought_match else None
    )
    clean_response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL | re.IGNORECASE,
    ).strip()

    if prompt_type == ExtractionPrompt.JSON:
        definitions = _parse_json_definitions(clean_response)
    elif prompt_type == ExtractionPrompt.STRUCTURED:
        definitions = _parse_structured_definitions(clean_response)
    else:
        definitions = _parse_standard_definitions(clean_response)

    return definitions, thought_process


def _parse_json_definitions(text: str) -> List[Definition]:
    """Parse JSON format definitions."""
    definitions = []

    # Try to extract JSON
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if not json_match:
        json_match = re.search(r"\[.*\]", text, re.DOTALL)

    if json_match:
        try:
            data = json.loads(json_match.group())

            if isinstance(data, dict):
                for concept, definition in data.items():
                    if concept and definition:
                        definitions.append(
                            Definition(
                                concept=str(concept),
                                definition_text=str(definition),
                            ),
                        )
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        concept = item.get("concept") or item.get("term")
                        definition = item.get("definition") or item.get(
                            "meaning",
                        )
                        if concept and definition:
                            definitions.append(
                                Definition(
                                    concept=str(concept),
                                    definition_text=str(definition),
                                ),
                            )

        except json.JSONDecodeError:
            return _parse_standard_definitions(text)

    return definitions if definitions else _parse_standard_definitions(text)


def _parse_structured_definitions(text: str) -> List[Definition]:
    """Parse structured format definitions."""
    definitions = []

    # Look for "Concept X:" followed by "Definition:"
    sections = re.split(
        r"\n\s*(?:Concept|Term|Definition)\s*(?:\d+)?\s*:",
        text,
        flags=re.IGNORECASE,
    )

    for i in range(1, len(sections), 2):
        if i + 1 < len(sections):
            concept = sections[i].strip()
            definition = sections[i + 1].strip()

            concept = re.sub(r"^[:\-\s]+", "", concept).strip()
            definition = re.sub(r"^[:\-\s]+", "", definition).strip()

            if concept and definition:
                definitions.append(
                    Definition(concept=concept, definition_text=definition),
                )

    return definitions if definitions else _parse_standard_definitions(text)


def _parse_standard_definitions(text: str) -> List[Definition]:
    """Parse common formats for definitions, including 'Concept - Definition' and 'Concept: Definition'."""
    definitions: List[Definition] = []

    # Try multiple patterns, ordered from most specific/common to more generic
    patterns = [
        # Plain lines or bullets: Concept - Definition
        re.compile(
            r"^\s*(?!Type:|Section:|Related:)\s*[-*•]?\s*(.+?)\s*-\s*(.+)$",
            re.MULTILINE,
        ),
        # Markdown bold: **Concept**: Definition
        re.compile(
            r"\*\*(.+?)\*\*\s*:\s*(.+?)(?=\n\s*\*\*|\n\s*$|\Z)",
            re.DOTALL,
        ),
        # Bulleted: - Concept: Definition
        re.compile(
            r"^\s*[-*•]\s*\*?\*?\s*(.+?)\*?\*?\s*:\s*(.+)",
            re.MULTILINE,
        ),
        # Numbered: 1. Concept: Definition
        re.compile(
            r"^\s*\d+\.\s*\*?\*?\s*(.+?)\*?\*?\s*:\s*(.+)",
            re.MULTILINE,
        ),
    ]

    for pattern in patterns:
        matches = pattern.findall(text)
        if matches:
            for concept, definition in matches:
                cleaned_concept = re.sub(r"[*_`]+", "", concept).strip()
                cleaned_definition = re.sub(r"[*_`]+", "", definition).strip()

                if (
                    cleaned_concept
                    and cleaned_definition
                    and len(cleaned_definition) > 10
                    and len(cleaned_concept) <= 120
                ):
                    definitions.append(
                        Definition(
                            concept=cleaned_concept,
                            definition_text=cleaned_definition,
                        ),
                    )

            if definitions:
                break

    return definitions
