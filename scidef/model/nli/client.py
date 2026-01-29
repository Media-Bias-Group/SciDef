import hashlib
import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scidef.model.dataclass import (
    CacheModel,
    ExtractionPrompt,
    NLIResult as EvalNLIResult,
)
from scidef.model.nli.dataclass import NLIMode, NLIResult, ScoreMode
from scidef.model.nli.utils import amean, hmean
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)

# Singleton storage for NLI models
_nli_models: Dict[str, Dict[str, Any]] = {}
_nli_init_lock = Lock()


class NLIClient(CacheModel):
    """Natural Language Inference client using transformers."""

    def __init__(
        self,
        model_name: str = "tasksource/ModernBERT-large-nli",
        device: str = "auto",
        dtype: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 2048,
        cache_dir: Optional[Path] = None,
        compile: bool = False,
    ):
        """Initialize the NLI client."""
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.compile = compile

        self.tokenizer = None
        self.model = None
        self.id2label = None
        self._initialized = False

        self._lock = Lock()

        # Ensure cache directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def initialize(self) -> None:
        """Initialize the NLI model and resources (singleton per model_name)."""
        if self._initialized:
            return

        global _nli_models

        with _nli_init_lock:
            # Check again inside lock (double-checked locking)
            if self._initialized:
                return

            # Check if model already loaded by another instance
            if self.model_name in _nli_models:
                cached = _nli_models[self.model_name]
                self.model = cached["model"]
                self.tokenizer = cached["tokenizer"]
                self.id2label = cached["id2label"]
                self.max_length = cached["max_length"]
                self._initialized = True
                logger.info(f"Reusing existing NLI model: {self.model_name}")
                return

            try:
                logger.info(f"Initializing NLI model: {self.model_name}")

                # Determine device
                if self.device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                elif self.device.startswith("cuda"):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = "cpu"  # CPU

                if device == "cuda":
                    # Enable TF32 for faster computation on Ampere+ GPUs
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True

                # Set torch dtype
                torch_dtype = None
                if self.dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif self.dtype == "float16":
                    torch_dtype = torch.float16
                elif self.dtype == "float32":
                    torch_dtype = torch.float32

                # Load tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        torch_dtype=torch_dtype,
                    ).to(device)
                )

                if self.model is not None:
                    self.max_length = self.model.config.max_position_embeddings

                if (
                    hasattr(self.model.config, "id2label")
                    and self.model.config.id2label
                ):
                    self.id2label = self.model.config.id2label
                    logger.info(
                        f"NLI model id2label: {self.model.config.id2label}",
                    )
                else:
                    self.id2label = {
                        0: "CONTRADICTION",
                        1: "ENTAILMENT",
                        2: "NEUTRAL",
                    }
                    logger.warning(
                        f"NLI model has no label configuration - using fallback mapping: {self.id2label}",
                    )

                self.model.eval()

                if self.compile:
                    self.model = torch.compile(self.model, mode="max-autotune")

                # Store in singleton cache
                _nli_models[self.model_name] = {
                    "model": self.model,
                    "tokenizer": self.tokenizer,
                    "id2label": self.id2label,
                    "max_length": self.max_length,
                }

                self._initialized = True
                logger.info(
                    f"NLI model initialized successfully on device: {device}",
                )

            except Exception as e:
                logger.error(f"Failed to initialize NLI model: {e}")
                raise

    async def run(
        self,
        human_definition: str,
        model_definition: str,
        paper_id: str,
        human_concept: str,
        model_mode: ExtractionPrompt,
        mode: NLIMode = NLIMode.FORWARD,
        score_mode: Union[List[ScoreMode], ScoreMode] = ScoreMode.HMEAN,
    ) -> NLIResult:
        try:
            if (
                len(human_definition) + len(model_definition) + 3
                > self.max_length
            ):
                logger.error(
                    "Input definitions exceed NLI's max length!",
                )
                raise RuntimeError("Input definitions exceed NLI's max length")

            if mode == NLIMode.FORWARD:
                result = self.evaluate_forward(
                    premise=human_definition,
                    hypothesis=model_definition,
                )
                if "error" in result:
                    raise RuntimeError(result["error"])

                return NLIResult(
                    paper_id=paper_id,
                    human_concept=human_concept,
                    human_definition=human_definition,
                    model_mode=model_mode,
                    model_definition=model_definition,
                    entailment_score=result["entailment_score"],
                    contradiction_score=result["contradiction_score"],
                    neutral_score=result["neutral_score"],
                    predicted_label=result["predicted_label"],
                    nli_model_name=self.model_name,
                )

            elif mode == NLIMode.BIDIRECTIONAL:
                result = self.evaluate_bidirectional(
                    premise=human_definition,
                    hypothesis=model_definition,
                    score_mode=score_mode,
                )

                return NLIResult(
                    paper_id=paper_id,
                    human_concept=human_concept,
                    human_definition=human_definition,
                    model_mode=model_mode,
                    model_definition=model_definition,
                    nli_model_name=self.model_name,
                    bidirectional_scores=result["bidirectional_scores"],
                    entailment_score=result["forward"]["entailment_score"],
                    contradiction_score=result["forward"][
                        "contradiction_score"
                    ],
                    neutral_score=result["forward"]["neutral_score"],
                    predicted_label=result["predicted_label"],
                    forward_entailment_score=result["forward"][
                        "entailment_score"
                    ],
                    backward_entailment_score=result["backward"][
                        "entailment_score"
                    ],
                    forward_predicted_label=result["forward"][
                        "predicted_label"
                    ],
                    backward_predicted_label=result["backward"][
                        "predicted_label"
                    ],
                    bidirectional_equivalent=result[
                        "bidirectional_equivalent"
                    ],
                )
            else:
                raise RuntimeError("Method not implemented")

        except Exception as e:
            return NLIResult(
                paper_id=paper_id,
                human_concept=human_concept,
                human_definition=human_definition,
                model_mode=model_mode,
                model_definition=model_definition,
                nli_error=str(e),
            )

    def _evaluate_nli_pair(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict[str, float]:
        """Evaluate NLI for a single premise-hypothesis pair."""
        # Cache
        prompts = (premise, hypothesis)
        cached_result = self.load_from_cache(
            prompts,
            model=self.model_name,
        )

        if cached_result is not None:
            return cached_result

        if not self._initialized:
            self.initialize()

        features = self.tokenizer(
            premise,
            hypothesis,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_overflowing_tokens=True,
        )  # type: ignore

        if len(features["overflow_to_sample_mapping"]) > 1:
            logger.error(
                f"[NLI] Input was truncated for NLI model '{self.model_name}'",
            )
            return {"error": -1.0}

        # Remove overflow_to_sample_mapping before passing to model
        features.pop("overflow_to_sample_mapping", None)

        # Move features to the same device as the model
        assert self.model is not None
        model_device = next(self.model.parameters()).device
        features = {k: v.to(model_device) for k, v in features.items()}

        with self._lock:
            with torch.no_grad():
                scores = self.model(**features).logits
                probabilities = torch.softmax(scores, dim=1)

                labels = {
                    self.id2label[i].upper(): float(probabilities[0][i])  # type: ignore
                    for i in range(len(self.id2label))  # type: ignore
                }
                self.save_to_cache(
                    prompts,
                    model=self.model_name,
                    result=labels,
                )
                return labels

    def evaluate_forward(
        self,
        premise: str,
        hypothesis: str,
    ) -> Dict:
        """Evaluate entailment between human and model definitions."""
        try:
            # Simple unidirectional entailment: human -> model
            scores = self._evaluate_nli_pair(premise, hypothesis)
            if "error" in scores:
                return {"error": scores["error"]}

            entailment_score = scores.get("ENTAILMENT", 0.0)
            contradiction_score = scores.get("CONTRADICTION", 0.0)
            neutral_score = scores.get("NEUTRAL", 0.0)

            # Determine predicted label
            predicted_label = max(scores, key=lambda k: scores[k])

            return {
                "predicted_label": predicted_label,
                "entailment_score": entailment_score,
                "contradiction_score": contradiction_score,
                "neutral_score": neutral_score,
            }
        except Exception as e:
            error_msg = f"NLI evaluation failed: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    def evaluate_bidirectional(
        self,
        premise: str,
        hypothesis: str,
        score_mode: Union[List[ScoreMode], ScoreMode],
    ) -> Dict:
        """Evaluate bidirectional entailment between human and model definitions."""
        if isinstance(score_mode, ScoreMode):
            score_mode = [score_mode]

        try:
            forward = self.evaluate_forward(
                premise=premise,
                hypothesis=hypothesis,
            )
            backward = self.evaluate_forward(
                premise=hypothesis,
                hypothesis=premise,
            )

            if "error" in forward or "error" in backward:
                error_msg = (
                    forward.get("error")
                    or backward.get("error")
                    or "Unknown error in bidirectional NLI evaluation"
                )
                raise RuntimeError(error_msg)

            bidirectional_scores = [-1.0, -1.0]
            if ScoreMode.HMEAN in score_mode:
                bidirectional_scores[0] = hmean(
                    forward["entailment_score"],
                    backward["entailment_score"],
                )
            if ScoreMode.AMEAN in score_mode:
                bidirectional_scores[1] = amean(
                    forward["entailment_score"],
                    backward["entailment_score"],
                )

            bidirectional_equivalent = (
                forward["predicted_label"] == "ENTAILMENT"
                and backward["predicted_label"] == "ENTAILMENT"
            )

            # Set predicted label based on bidirectional analysis
            if bidirectional_equivalent:
                predicted_label = "bidirectional_entailment"
            elif forward["predicted_label"] == "ENTAILMENT":
                predicted_label = "unidirectional_entailment"
            else:
                predicted_label = forward["predicted_label"].lower()

            return {
                "bidirectional_scores": bidirectional_scores,
                "predicted_label": predicted_label,
                "bidirectional_equivalent": bidirectional_equivalent,
                "forward": forward,
                "backward": backward,
            }

        except Exception as e:
            error_msg = f"Bidirectional NLI evaluation failed: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

    def evaluate_bidirectional_entailment(
        self,
        paper_id: str,
        human_concept: str,
        human_definition: str,
        model_mode: ExtractionPrompt,
        model_definition: str,
        score_mode: Union[List[ScoreMode], ScoreMode] = ScoreMode.HMEAN,
    ) -> EvalNLIResult:
        """Return an evaluation-friendly NLIResult for bidirectional scoring."""
        result = self.evaluate_bidirectional(
            premise=human_definition,
            hypothesis=model_definition,
            score_mode=score_mode,
        )
        if "error" in result:
            return EvalNLIResult(
                paper_id=paper_id,
                human_concept=human_concept,
                human_definition=human_definition,
                model_mode=model_mode,
                model_definition=model_definition,
                nli_model_name=self.model_name,
                nli_error=str(result["error"]),
            )
        forward = result["forward"]
        backward = result["backward"]
        return EvalNLIResult(
            paper_id=paper_id,
            human_concept=human_concept,
            human_definition=human_definition,
            model_mode=model_mode,
            model_definition=model_definition,
            entailment_score=forward.get("entailment_score"),
            contradiction_score=forward.get("contradiction_score"),
            neutral_score=forward.get("neutral_score"),
            predicted_label=result.get("predicted_label"),
            nli_model_name=self.model_name,
            forward_entailment_score=forward.get("entailment_score"),
            backward_entailment_score=backward.get("entailment_score"),
            forward_predicted_label=forward.get("predicted_label"),
            backward_predicted_label=backward.get("predicted_label"),
            bidirectional_equivalent=result.get("bidirectional_equivalent"),
        )

    def is_available(self) -> bool:
        """Check if NLI service is available."""
        return self._initialized

    def get_cache_key(
        self,
        prompts: tuple[str, str],
        model: str,
        **kwargs: Any,
    ) -> str:
        """Generate cache key for NLI forward parameters."""
        params: Dict[str, Any] = {
            "model": model,
            "prompts": prompts,
        }

        for key, value in kwargs.items():
            if value is not None:
                params[key] = value

        content = json.dumps(params, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def load_from_cache(
        self,
        prompts: tuple[str, str],
        model: str,
        **kwargs: Any,
    ) -> Optional[Dict[str, float]]:
        """Load NLI classification from cache if available."""
        if not self.cache_dir:
            return None

        try:
            cache_key = self.get_cache_key(
                prompts,
                model,
                **kwargs,
            )
            cache_file = self.cache_dir / f"{cache_key}.json"

            if cache_file.exists():
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                if (
                    "result" in cache_data
                    and "model" in cache_data
                    and cache_data["model"] == model
                ):
                    return cache_data["result"]

        except Exception as e:
            logger.debug(f"Error loading NLI response from cache: {e}")

        return None

    def save_to_cache(
        self,
        prompts: tuple[str, str],
        model: str,
        result: Dict[str, float],
        **kwargs: Any,
    ) -> None:
        """Save NLI response to cache."""
        if not self.cache_dir:
            return

        try:
            cache_key = self.get_cache_key(
                prompts,
                model,
                **kwargs,
            )
            cache_file = self.cache_dir / f"{cache_key}.json"

            cache_data = {
                "prompts": prompts,
                "model": model,
                "result": result,
            }

            for key, value in kwargs.items():
                if value is not None and key not in cache_data:
                    cache_data[f"param_{key}"] = value

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

        except Exception as e:
            logger.debug(f"Error saving NLI response to cache: {e}")
