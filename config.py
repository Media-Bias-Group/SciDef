"""
Configuration and service factory for the simplified definition extraction system.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from scidef.model.dataclass import (
    ConfigurationError,
    ExtractionPrompt,
    ProcessingConfiguration,
    ProviderType,
)
from scidef.model.embedding.client import EmbeddingClient
from scidef.model.llm.client import LLMClient
from scidef.model.llm.judge.client import JudgeClient
from scidef.model.nli.client import NLIClient
from scidef.utils import (
    generate_performance_summary,
    generate_similarity_histogram,
    get_custom_colored_logger,
    save_evaluation_results_to_csv,
    save_extraction_results_to_csv,
)

logger = get_custom_colored_logger(__name__)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup basic logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def get_csv_params() -> dict:
    """Get CSV parameters for consistent file handling."""
    return {
        "sep": os.getenv("CSV_DELIMITER", "|"),
        "quotechar": os.getenv("CSV_QUOTECHAR", '"'),
    }


class Config:
    """Main configuration class for the definition extraction system."""

    def __init__(self, env_file: Optional[Path] = None):
        self.project_root = Path(__file__).parent

        # environment variables
        if env_file and env_file.exists():
            load_dotenv(env_file)
        elif (self.project_root / ".env").exists():
            load_dotenv(self.project_root / ".env")

        # Load configuration
        self.processing_config = self._load_processing_config()
        self.paths = self._setup_paths()

        # Ensure directories exist
        for path in [
            self.paths["output_dir"],
            self.paths["cache_dir"],
            self.paths["logs_dir"],
        ]:
            path.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Configuration loaded: {self.processing_config.llm_provider.value} LLM provider",
        )

    def _load_processing_config(self) -> ProcessingConfiguration:
        """Load processing configuration from environment."""
        try:
            # LLM Configuration
            llm_provider_str = os.getenv("LLM_PROVIDER", "openrouter").lower()
            try:
                llm_provider = ProviderType(llm_provider_str)
            except ValueError:
                raise ConfigurationError(
                    f"Invalid LLM provider: {llm_provider_str}",
                )

            llm_model_name = os.getenv("LLM_MODEL_NAME", "")
            if llm_provider == ProviderType.VLLM:
                llm_base_url = os.getenv(
                    "VLLM_BASE_URL",
                    "http://localhost:8000/v1",
                )
                llm_api_key = os.getenv("VLLM_API_KEY", "EMPTY")
                if not llm_model_name:
                    llm_model_name = os.getenv("VLLM_MODEL_NAME", "")
            else:  # OpenRouter
                llm_base_url = os.getenv(
                    "OPENROUTER_BASE_URL",
                    "https://openrouter.ai/api/v1",
                )
                llm_api_key = os.getenv("OPENROUTER_API_KEY", "")
                if not llm_model_name:
                    llm_model_name = os.getenv(
                        "OPENROUTER_MODEL_NAME",
                        "meta-llama/llama-3.3-8b-instruct:free",
                    )

            # Embedding Configuration (optional)
            embedding_provider_str = os.getenv(
                "EMBEDDING_PROVIDER",
                "",
            ).lower()
            embedding_provider = None
            embedding_base_url = None
            embedding_api_key = None
            embedding_model_name = None

            if embedding_provider_str:
                try:
                    embedding_provider = ProviderType(embedding_provider_str)
                    if embedding_provider == ProviderType.VLLM:
                        embedding_base_url = os.getenv(
                            "VLLM_EMBEDDING_BASE_URL",
                            "http://localhost:8001/v1",
                        )
                        embedding_api_key = os.getenv(
                            "VLLM_EMBEDDING_API_KEY",
                            "EMPTY",
                        )
                        embedding_model_name = os.getenv(
                            "VLLM_EMBEDDING_MODEL_NAME",
                            "",
                        )
                    else:  # OpenRouter
                        embedding_base_url = llm_base_url
                        embedding_api_key = llm_api_key
                        embedding_model_name = os.getenv(
                            "OPENROUTER_EMBEDDING_MODEL_NAME",
                            "",
                        )
                except ValueError:
                    logger.warning(
                        f"Invalid embedding provider: {embedding_provider_str}, disabling embeddings",
                    )

            # Processing modes
            processing_modes_str = os.getenv(
                "PROCESSING_MODES",
                "json,extractive",
            ).lower()
            processing_modes = []
            for mode_str in processing_modes_str.split(","):
                mode_str = mode_str.strip()
                try:
                    processing_modes.append(ExtractionPrompt(mode_str))
                except ValueError:
                    logger.warning(
                        f"Invalid processing mode: {mode_str}, skipping",
                    )

            if not processing_modes:
                processing_modes = [
                    ExtractionPrompt.JSON,
                    ExtractionPrompt.EXTRACTIVE,
                ]

            # Other parameters
            max_concurrency = int(os.getenv("ASYNC_MAX_CONCURRENCY", "1"))
            min_article_words = int(os.getenv("MIN_ARTICLE_WORDS", "50"))
            max_article_words = os.getenv("MAX_ARTICLE_WORDS", None)
            max_article_words = (
                int(max_article_words) if max_article_words else None
            )
            max_retries = int(os.getenv("MAX_RETRIES", "3"))
            timeout = float(os.getenv("TIMEOUT", "600.0"))

            # Chunking for long articles
            chunk_words = os.getenv("CHUNK_WORDS", None)
            chunk_words = int(chunk_words) if chunk_words else None
            chunk_overlap_words = int(os.getenv("CHUNK_OVERLAP_WORDS", "200"))
            max_chunks_per_paper = int(os.getenv("MAX_CHUNKS_PER_PAPER", "10"))

            # Validate required fields
            if not llm_model_name:
                raise ConfigurationError("LLM model name not configured")
            if llm_provider == ProviderType.OPENROUTER and (
                not llm_api_key or llm_api_key == "EMPTY"
            ):
                raise ConfigurationError("OpenRouter API key not configured")

            return ProcessingConfiguration(
                llm_provider=llm_provider,
                llm_model_name=llm_model_name,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedding_provider=embedding_provider,
                embedding_model_name=embedding_model_name,
                embedding_base_url=embedding_base_url,
                embedding_api_key=embedding_api_key,
                processing_modes=processing_modes,
                max_concurrency=max_concurrency,
                min_article_words=min_article_words,
                max_article_words=max_article_words,
                chunk_words=chunk_words,
                chunk_overlap_words=chunk_overlap_words,
                max_chunks_per_paper=max_chunks_per_paper,
                max_retries=max_retries,
                timeout=timeout,
            )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration: {e}",
            ) from e

    def _setup_paths(self) -> dict:
        """Setup directory paths."""
        xml_input_dir = Path(
            os.getenv(
                "XML_INPUT_DIR",
                self.project_root / "ManualPDFsGROBID" / "manual_pdfs_grobid",
            ),
        )
        output_dir = Path(
            os.getenv("OUTPUT_DIR", self.project_root / "results"),
        )
        cache_dir = Path(os.getenv("CACHE_DIR", output_dir / ".cache"))
        logs_dir = Path(os.getenv("LOGS_DIR", output_dir / "logs"))

        return {
            "project_root": self.project_root,
            "xml_input_dir": xml_input_dir,
            "output_dir": output_dir,
            "cache_dir": cache_dir,
            "logs_dir": logs_dir,
        }

    def get_min_word_count(self) -> int:
        """Get minimum word count for text extraction."""
        return self.processing_config.min_article_words

    def create_llm_client(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        disable_cache: bool = False,
        api_key: Optional[str] = None,
        log_level: int = logging.INFO,
    ) -> LLMClient:
        """Create LLM client service."""
        return LLMClient(
            provider_type=self.processing_config.llm_provider,
            base_url=base_url or self.processing_config.llm_base_url,
            api_key=api_key or self.processing_config.llm_api_key,
            model_name=model_name or self.processing_config.llm_model_name,
            timeout=self.processing_config.timeout,
            max_retries=self.processing_config.max_retries,
            cache_dir=cache_dir or None
            if disable_cache
            else self.paths["cache_dir"] / "llm_responses",
            log_level=log_level,
        )

    def create_judge_client(self) -> JudgeClient:
        """Create Judge LLM client service."""
        return JudgeClient(
            provider_type=self.processing_config.llm_provider,
            base_url=self.processing_config.llm_base_url,
            api_key=self.processing_config.llm_api_key,
            model_name=self.processing_config.llm_model_name,
            timeout=self.processing_config.timeout,
            max_retries=self.processing_config.max_retries,
            cache_dir=self.paths["cache_dir"] / "llm_responses",
        )

    def create_embedding_client(self) -> Optional[EmbeddingClient]:
        """Create embedding client service."""
        if not all(
            [
                self.processing_config.embedding_provider,
                self.processing_config.embedding_base_url,
                self.processing_config.embedding_api_key,
                self.processing_config.embedding_model_name,
            ],
        ):
            return None

        return EmbeddingClient(
            base_url=self.processing_config.embedding_base_url or "",
            api_key=self.processing_config.embedding_api_key or "",
            model_name=self.processing_config.embedding_model_name or "",
            cache_dir=self.paths["cache_dir"] / "embeddings",
        )

    def create_nli_client(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        cache_dir: Optional[Path] = None,
        compile: Optional[bool] = None,
    ) -> Optional[NLIClient]:
        """Create NLI client service."""
        try:
            return NLIClient(
                model_name=model_name or self.processing_config.nli_model_name,
                device=device or self.processing_config.nli_device,
                dtype=dtype or self.processing_config.nli_dtype,
                batch_size=batch_size or self.processing_config.nli_batch_size,
                max_length=max_length or self.processing_config.nli_max_length,
                cache_dir=cache_dir or self.paths["cache_dir"] / "nli_models",
                compile=compile or self.processing_config.nli_compile,
            )
        except Exception as e:
            logger.warning(f"Failed to create NLI client: {e}")
            return None

    def create_llm_judge(
        self,
    ) -> Optional[JudgeClient]:
        """Create LLM judge service."""
        return self.create_judge_client()

    def create_result_storage(self):
        """Create result storage functions."""
        return {
            "save_extraction_results": save_extraction_results_to_csv,
            "save_evaluation_results": save_evaluation_results_to_csv,
        }

    def create_visualization_service(self):
        """Create visualization functions."""
        return {
            "generate_similarity_histogram": generate_similarity_histogram,
            "generate_performance_summary": generate_performance_summary,
        }

    def get_model_suffix(self) -> str:
        """Get model suffix for output file naming."""
        model_name = self.processing_config.llm_model_name
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        return model_name.replace(":", "_")

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of warnings."""
        messages = []

        # Check if XML input directory exists
        if not self.paths["xml_input_dir"].exists():
            messages.append(
                f"XML input directory does not exist: {self.paths['xml_input_dir']}",
            )

        # Check for placeholder values
        if "your-" in self.processing_config.llm_api_key:
            messages.append("LLM API key appears to be a placeholder")

        # Check concurrency settings for OpenRouter free tier
        if (
            self.processing_config.llm_provider == ProviderType.OPENROUTER
            and ":free" in self.processing_config.llm_model_name
            and self.processing_config.max_concurrency > 2
        ):
            messages.append(
                f"High concurrency ({self.processing_config.max_concurrency}) detected for "
                "OpenRouter free tier - may hit rate limits",
            )

        return messages
