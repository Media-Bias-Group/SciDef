import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


class CacheModel(ABC):
    """Abstract class for models that allow caching"""

    @abstractmethod
    def get_cache_key(self, *args, **kwargs) -> str:
        raise NotImplementedError("get_cache_key not implemented!")

    @abstractmethod
    def load_from_cache(self, *args, **kwargs) -> Optional[Any]:
        raise NotImplementedError("load_from_cache not implemented!")

    @abstractmethod
    def save_to_cache(self, *args, **kwargs) -> None:
        raise NotImplementedError("save_to_cache not implemented!")


class JudgeSystemPrompt(Enum):
    BINARY = "binary"
    TERNARY = "ternary"
    CATEGORICAL4 = "categorical4"


class JudgeCategory(Enum):
    """4-point monotone scale for definition equivalence judgments."""

    DIFFERENT = 0  # Contradictory, different referent, or no substantive overlap
    RELATED = 1  # Overlap exists but not equivalent
    NEAR_SAME = 2  # Very close; minor omissions or wording differences
    SAME = 3  # Same concept with equivalent necessary & sufficient conditions


class ExtractionPrompt(Enum):
    """Different prompts for definition extraction."""

    EXTRACTIVE = "extractive"
    STRUCTURED = "structured"
    JSON = "json"


class ProcessingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProviderType(Enum):
    VLLM = "vllm"
    OPENROUTER = "openrouter"


class EvaluationMetric(Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    LLM_JUDGE = "llm_judge"
    NLI = "nli"
    BOTH = "both"
    ALL = "all"


@dataclass
class ProcessingMetrics:
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    llm_request_duration: float = 0.0
    prompt_processing_duration: float = 0.0
    total_duration: float = 0.0

    def mark_complete(self) -> None:
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time

    @property
    def is_complete(self) -> bool:
        return self.end_time is not None


@dataclass
class Definition:
    concept: str
    definition_text: str
    confidence: Optional[float] = None
    source_section: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.concept.strip():
            raise ValueError("Concept cannot be empty")
        if not self.definition_text.strip():
            raise ValueError("Definition text cannot be empty")


@dataclass
class ExtractionResult:
    paper_id: str
    mode: ExtractionPrompt
    definitions: List[Definition] = field(default_factory=list)
    raw_llm_output: Optional[str] = None
    thought_process: Optional[str] = None
    error_message: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("Paper ID cannot be empty")

    @property
    def is_successful(self) -> bool:
        return (
            self.status == ProcessingStatus.COMPLETED
            and not self.error_message
            and len(self.definitions) > 0
        )

    def mark_failed(self, error: str) -> None:
        self.status = ProcessingStatus.FAILED
        self.error_message = error
        self.metrics.mark_complete()

    def mark_completed(self) -> None:
        self.status = ProcessingStatus.COMPLETED
        self.metrics.mark_complete()


@dataclass
class PaperMetadata:
    paper_id: str
    xml_file_path: Path
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    word_count: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("Paper ID cannot be empty")
        if not self.xml_file_path.exists():
            raise ValueError(f"XML file does not exist: {self.xml_file_path}")


@dataclass
class HumanAnnotation:
    paper_id: str
    defined_concepts: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
    raw_concept_text: Optional[str] = None
    raw_definition_text: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.paper_id.strip():
            raise ValueError("Paper ID cannot be empty")
        # Note: Relaxed validation for testing - concepts and definitions may not match exactly


@dataclass
class SimilarityResult:
    paper_id: str
    human_concept: str
    human_definition: str
    model_mode: ExtractionPrompt
    model_definition: str
    similarity_score: Optional[float] = None
    embedding_error: Optional[str] = None

    @property
    def is_valid_comparison(self) -> bool:
        return self.similarity_score is not None and not self.embedding_error


@dataclass
class JudgeResult:
    paper_id: str
    human_concept: str
    human_definition: str
    model_mode: ExtractionPrompt
    model_definition: str
    judgment_category: Optional[Union[int, float]] = None
    judge_prompt_template: Optional[str] = None
    judgment_confidence: Optional[float] = None
    judgment_error: Optional[str] = None

    @property
    def is_valid_judgment(self) -> bool:
        return self.judgment_category is not None and not self.judgment_error


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

    # Bidirectional entailment metadata
    forward_entailment_score: Optional[float] = None
    backward_entailment_score: Optional[float] = None
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


@dataclass
class EvaluationResult:
    paper_id: str
    human_annotations: HumanAnnotation
    extraction_results: List[ExtractionResult] = field(default_factory=list)
    similarity_results: List[SimilarityResult] = field(default_factory=list)
    judge_results: List[JudgeResult] = field(default_factory=list)
    nli_results: List[NLIResult] = field(default_factory=list)

    @property
    def successful_extractions(self) -> List[ExtractionResult]:
        return [r for r in self.extraction_results if r.is_successful]

    @property
    def valid_similarities(self) -> List[SimilarityResult]:
        return [r for r in self.similarity_results if r.is_valid_comparison]

    @property
    def valid_judgments(self) -> List[JudgeResult]:
        return [r for r in self.judge_results if r.is_valid_judgment]

    @property
    def valid_nli_results(self) -> List[NLIResult]:
        return [r for r in self.nli_results if r.is_valid_nli]


@dataclass
class ProcessingConfiguration:
    # LLM Configuration
    llm_provider: ProviderType
    llm_model_name: str
    llm_base_url: str
    llm_api_key: str

    # Judge LLM Configuration (optional)
    judge_llm_provider: Optional[ProviderType] = None
    judge_llm_model_name: Optional[str] = None
    judge_llm_base_url: Optional[str] = None
    judge_llm_api_key: Optional[str] = None

    # Embedding Configuration
    embedding_provider: Optional[ProviderType] = None
    embedding_model_name: Optional[str] = None
    embedding_base_url: Optional[str] = None
    embedding_api_key: Optional[str] = None

    # NLI Configuration
    nli_model_name: str = "roberta-large-mnli"
    nli_device: str = "auto"
    nli_dtype: Optional[str] = None
    nli_batch_size: int = 8
    nli_max_length: int = 512
    nli_compile: bool = False

    # Processing Parameters
    processing_modes: List[ExtractionPrompt] = field(
        default_factory=lambda: [
            ExtractionPrompt.JSON,
            ExtractionPrompt.EXTRACTIVE,
        ],
    )
    max_concurrency: int = 1
    min_article_words: int = 50
    max_article_words: Optional[int] = None
    chunk_words: Optional[int] = None
    chunk_overlap_words: int = 200
    max_chunks_per_paper: int = 8
    use_all_prompt_variations: bool = True

    # Retry Configuration
    max_retries: int = 3
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not self.llm_model_name.strip():
            raise ValueError("LLM model name cannot be empty")
        if not self.llm_base_url.strip():
            raise ValueError("LLM base URL cannot be empty")
        if self.max_concurrency < 1:
            raise ValueError("Max concurrency must be at least 1")
        if self.min_article_words < 1:
            raise ValueError("Min article words must be at least 1")

    def get_judge_llm_config(self) -> Tuple[ProviderType, str, str, str]:
        provider = self.judge_llm_provider or self.llm_provider
        model_name = self.judge_llm_model_name or self.llm_model_name
        base_url = self.judge_llm_base_url or self.llm_base_url
        api_key = self.judge_llm_api_key or self.llm_api_key
        return provider, model_name, base_url, api_key


# Custom exceptions
class DefinitionExtractionError(Exception):
    def __init__(
        self,
        message: str,
        paper_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.paper_id = paper_id
        self.details = details or {}

    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.paper_id:
            base_msg = f"[Paper: {self.paper_id}] {base_msg}"
        if self.details:
            details_str = ", ".join(
                f"{k}={v}" for k, v in self.details.items()
            )
            base_msg = f"{base_msg} (Details: {details_str})"
        return base_msg


class TextExtractionError(DefinitionExtractionError):
    pass


class LLMClientError(DefinitionExtractionError):
    def __init__(
        self,
        message: str,
        paper_id: Optional[str] = None,
        status_code: Optional[int] = None,
        retry_count: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        details = details or {}
        if status_code:
            details["status_code"] = status_code
        if retry_count:
            details["retry_count"] = retry_count
        super().__init__(message, paper_id, details)
        self.status_code = status_code
        self.retry_count = retry_count


class EmbeddingError(DefinitionExtractionError):
    pass


class ConfigurationError(DefinitionExtractionError):
    pass


class ProcessingError(DefinitionExtractionError):
    pass


class ServiceUnavailableError(DefinitionExtractionError):
    pass


class EmptyResponseError(DefinitionExtractionError):
    """Raised when API returns an empty response - this should be retried."""

    pass
