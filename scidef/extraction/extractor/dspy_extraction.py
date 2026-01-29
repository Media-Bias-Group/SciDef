import json
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List

import dspy
from pydantic import BaseModel, Field

from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)

_TERM_WS = re.compile(r"\s+")
_TERM_PUN = re.compile(r"[()\[\]\"'`]")
_TERM_DASH = re.compile(r"[\/–—\-]+")


def normalize_term(s) -> str:
    """Normalize a term for comparison. Handles None and non-string inputs."""
    if s is None:
        return ""
    if not isinstance(s, str):
        # Handle lists, dicts, etc.
        if isinstance(s, (list, tuple)):
            logger.warning(
                f"Warning: term expected to be string, got list/tuple: {s}. Using first element.",
            )
            s = str(s[0]) if s else ""
        elif isinstance(s, dict):
            logger.warning(
                f"Warning: term expected to be string, got dict: {s}. Using 'term', 'name', or 'value' field if present.",
            )
            s = str(s.get("term", s.get("name", s.get("value", ""))))
        else:
            s = str(s)
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = _TERM_PUN.sub("", s)
    s = _TERM_DASH.sub(" ", s)
    s = _TERM_WS.sub(" ", s)
    return s


class TermExtraction(BaseModel):
    term: str = Field(description="The term being defined.")
    definition: str = Field(
        description="The explicit definition found in the text.",
    )
    context: str = Field(
        description="The verbatim sentences surrounding the definition (1 sentence before the definition, during, 1 sentence after the definition).",
    )
    type: str = Field(
        description="Type of definition - either 'explicit' or 'implicit'. Explicit is when the term is directly defined, implicit is when the definition is implied but not directly stated and/or the definition is heavily context dependent (and not useful outside of it) and/or it is edge of being a description, rather than a definition.",
    )


# ---- Signature: one input ('section'), one output ('extractions_json')
class ExtractPairsSig(dspy.Signature):
    """Extract (term, definition) pairs present in this section.
    Return a dictionary of definitions, e.g. {"hate speech": ["language that is used to expresses hatred towards a targeted group or is intended to be derogatory, to humiliate, or to insult the members of the group.", "context sentences..."], ...}.
    If none found, return {}.
    Rules:
    - Prefer definitions stated or strongly implied in this section.
    - Do not over-generate: only extract what is clearly defined in the section and what is a clearly a definition, not description, explanation, effect, or other information.
    - Each term must be explicitly defined in the section.
    - Do not hallucinate outside this section.
    - Each definition must be ideally 1 sentence long.
    - Remove the prefixes like "<term> is defined as" from the definition text and keep lowercase.
    - If multiple definitions are present, extract each one separately.
    - Unless absolutely certain, prefer returning no definitions to false positives.
    - Unless strongly required, copy the definition word by word from the source text!
    - If term has synonyms defined (not abbreviations!), divide them with '/' in the 'term' field.
    - For context, include 1 sentence before and 1 sentence after the definition sentence, if possible and don't change any words or formatting.
    """

    section = dspy.InputField(desc="Section of a paper to analyze.")
    extracted_terms: List[TermExtraction] = dspy.OutputField(
        desc="[Term, Definition, Context, Type] tuples extracted from the section. Context sentences where the definition was found (i.e., 1 previous sentence, the definition sentence, and 1 next sentence; if possible).",
    )


class DetermineIfDefinitionSig(dspy.Signature):
    """Determine if the given text contains a definition of a term.
    Rules:
    - A definition typically includes a clear explanation of a term's meaning.
    - Look for cue phrases like 'is defined as', 'refers to', or 'means'.
    - If unsure, prefer 'no' to avoid false positives.
    """

    section = dspy.InputField(desc="Section of a paper to analyze.")
    is_definition: bool = dspy.OutputField(
        desc="Does this section contain an obvious definition?",
    )


class ExtractFromSection(dspy.Module):
    def __init__(self):
        super().__init__()
        self.step = dspy.ChainOfThought(ExtractPairsSig)

    def forward(self, section: str):
        # Ensure valid JSON; if bad JSON, coerce to empty.
        out = self.step(section=section)

        # sanitize fields
        cleaned = []
        for term_extraction in out.extracted_terms:
            term = normalize_term(str(term_extraction.term).strip())
            definition = str(term_extraction.definition).strip()
            context = str(term_extraction.context).strip()
            type_ = str(term_extraction.type).strip()

            if type_ not in {"explicit", "implicit"}:
                logger.warning(
                    f"CRITICAL WARNING: Invalid type for term '{term}': type='{type_}'!!!!!!!!!!!!!!!!!!!!!",
                )
                logger.warning("--- \n" * 3)
            if context == "":
                logger.warning(
                    f"CRITICAL WARNING: Missing context for term '{term}': context='{context}'!!!!!!!!!!!!!!!!!!!!!",
                )
                logger.warning("--- \n" * 3)
            if definition == "":
                logger.warning(
                    f"CRITICAL WARNING: Missing definition for term '{term}': definition='{definition}'!!!!!!!!!!!!!!!!!!!!!",
                )
                logger.warning("--- \n" * 3)
            if term == "":
                logger.warning(
                    f"CRITICAL WARNING: Missing term: term='{term}'!!!!!!!!!!!!!!!!!!!!!",
                )
                logger.warning("--- \n" * 3)
            if (
                term
                and definition
                and context
                and type_ in {"explicit", "implicit"}
            ):
                cleaned.append(
                    {
                        "term": term,
                        "definition": definition,
                        "context": context,
                        "type": type_,
                    },
                )

        return dspy.Prediction(
            extractions_json=json.dumps(cleaned, ensure_ascii=False),
        )


class DSPyPaperExtractor(dspy.Module):
    def __init__(self, two_step: bool = False):
        super().__init__()
        self.extract = ExtractFromSection()
        self.two_step = two_step
        if self.two_step:
            self.determine = dspy.ChainOfThought(DetermineIfDefinitionSig)

    def forward(self, sections: List[str]):
        # MAP
        all_pairs = []
        for sec in sections:
            if self.two_step:
                is_def = self.determine(section=sec)
                if not is_def.is_definition:
                    continue
            res = self.extract(section=sec)
            pairs = json.loads(res.extractions_json)
            all_pairs.extend(pairs)

        # MERGE/DEDUP by term
        by_term: Dict[str, List[List[str]]] = defaultdict(list)
        for p in all_pairs:
            t = p["term"]
            d = p["definition"]
            ctx = p["context"]
            type_ = p["type"]
            by_term[t].append([d, ctx, type_])

        # TODO: pick the "best" definition per term?
        merged = []
        for t, defs in by_term.items():
            merged.append(
                {
                    "term": t,
                    "definition": defs[0][0],
                    "context": defs[0][1],
                    "type": defs[0][2],
                },
            )
        return dspy.Prediction(
            merged_json=json.dumps(merged, ensure_ascii=False),
        )
