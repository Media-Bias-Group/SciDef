from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from scidef.model.dataclass import ExtractionPrompt


class NLIMode(Enum):
    FORWARD = "forward"
    BIDIRECTIONAL = "bidirectional"


class ScoreMode(Enum):
    HMEAN = "hmean"
    AMEAN = "amean"


@dataclass
class NLIResult:
    paper_id: str
    human_concept: str
    human_definition: str
    model_mode: ExtractionPrompt
    model_definition: str
    entailment_score: Optional[float] = None
    contradiction_score: Optional[float] = None
    neutral_score: Optional[float] = None
    predicted_label: Optional[str] = None
    nli_model_name: Optional[str] = None
    nli_error: Optional[str] = None

    bidirectional_scores: Optional[List[float]] = None
    forward_entailment_score: Optional[float] = None
    forward_contradiction_score: Optional[float] = None
    forward_neutral_score: Optional[float] = None

    backward_entailment_score: Optional[float] = None
    backward_contradiction_score: Optional[float] = None
    backward_neutral_score: Optional[float] = None

    forward_predicted_label: Optional[str] = None
    backward_predicted_label: Optional[str] = None
    bidirectional_equivalent: Optional[bool] = None

    @property
    def is_valid_nli(self) -> bool:
        return (
            self.entailment_score is not None
            and self.predicted_label is not None
            and not self.nli_error
        )
