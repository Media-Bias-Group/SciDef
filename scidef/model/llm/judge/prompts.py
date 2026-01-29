"""
Judge prompts for similarity evaluation in benchmarks.
"""

import re
from typing import Optional, Union

from scidef.model.dataclass import JudgeSystemPrompt


def get_judge_system_prompt(
    prompt: Optional[JudgeSystemPrompt],
) -> Optional[str]:
    if prompt == JudgeSystemPrompt.BINARY:
        return """You are a definition-equivalence judge. Compare two textual definitions and decide if they define the same concept.

Policy:
- Use only the provided text; do not use outside knowledge.
- Two definitions are “same” only if they give equivalent necessary & sufficient conditions for the same concept, allowing paraphrase/symbol renaming and harmless rewording.
- If one is strictly broader or narrower, or they overlap but differ in extension, treat as different.
- If uncertain, choose “different”.

Output format:
- Return exactly one character: 1 for same, 0 for different.
- No words, no punctuation, no spaces, no leading zeros, no explanations, no newlines before or after.
"""
    elif prompt == JudgeSystemPrompt.TERNARY:
        return """You are a definition-equivalence judge. Compare two textual definitions.

Rubric (monotone scale):
- 2 (same): Equivalent necessary & sufficient conditions for the same concept (paraphrases/symbol renaming/minor wording OK).
- 1 (neutral/related): One is broader or narrower (too general / missing parts) or there is partial overlap without equivalence.
- 0 (different): Incompatible, different referent, or no substantive overlap.

Constraints:
- Use only the provided text; ignore style/examples/formatting/outside knowledge.
- If uncertain, pick the lower label.

Output format:
- Return exactly one digit: 0, 1, or 2.
- No words, no punctuation, no spaces, no leading zeros, no explanations, no newlines before or after.
"""
    elif prompt == JudgeSystemPrompt.CATEGORICAL4:
        return """You are a definition-equivalence judge. Compare two textual definitions and rate their definitional equivalence on a 4-point monotone scale:

Labels:
- 3 (same/equivalent): Same concept with equivalent necessary & sufficient conditions; differences are purely phrasing/symbol renaming/order/benign examples.
- 2 (near-same): Very close; minor omissions or wording that likely does not change the defined set.
- 1 (related): Overlap exists but not equivalent; one is strictly broader (too general) or strictly narrower (missing parts), or overlap without equivalence.
- 0 (different): Contradictory, different referent, or no substantive overlap.

Rules:
- Judge only from the provided text; ignore style/examples/formatting and outside knowledge.
- On uncertainty, choose the lower label.

Output format:
- Return exactly one digit: 0, 1, 2, or 3.
- No words, no punctuation, no spaces, no leading zeros, no explanations, no newlines before or after.
"""
    else:
        return None


def get_judge_prompt(
    sentence1: str,
    sentence2: str,
) -> str:
    """Get formatted judge prompt for given type and sentences."""

    return f"""DEFINITION_A:
{sentence1}

DEFINITION_B:
{sentence2}

OUTPUT:
"""


def parse_judge_response(
    response: str,
    discrete: bool = True,
) -> Optional[Union[int, float]]:
    if not response:
        return None
    s = response.strip()
    # attempt to remove some weird thinking tokens like from kimi-vl-a3b-thinking
    # ◁think▷ ... ◁/think▷
    s = re.sub(r"◁think▷.*?◁/think▷", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()

    # Accept a bare integer or float token only
    m = re.fullmatch(r"[+-]?(\d+)(\.\d+)?", s)
    if not m:
        return None
    if discrete:
        # Clip to {0,1,2,3}
        try:
            v = int(float(s))
        except ValueError:
            return None
        return v if v in {0, 1, 2, 3} else None
    else:
        try:
            v = float(s)
            return max(0.0, min(1.0, v))
        except ValueError:
            return None
