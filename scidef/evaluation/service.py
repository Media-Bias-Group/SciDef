"""
Evaluation services: embeddings, NLI, and comprehensive evaluation orchestration.
"""

from pathlib import Path
from typing import Dict, List, Optional, cast

import numpy as np

from scidef.model.dataclass import (
    EmbeddingError,
    EvaluationMetric,
    EvaluationResult,
    ExtractionPrompt,
    HumanAnnotation,
    JudgeCategory,
    JudgeResult,
    NLIResult,
    PaperMetadata,
    SimilarityResult,
)
from scidef.model.embedding.client import EmbeddingClient
from scidef.model.nli.client import NLIClient
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


class EvaluationService:
    """Main evaluation service that orchestrates all evaluation methods."""

    def __init__(
        self,
        definition_extractor,
        embedding_client: Optional[EmbeddingClient] = None,
        llm_judge=None,
        nli_client: Optional[NLIClient] = None,
        error_log_path: Optional[Path] = None,
    ):
        self.definition_extractor = definition_extractor
        self.embedding_client = embedding_client
        self.llm_judge = llm_judge
        self.nli_client = nli_client
        self.error_log_path = error_log_path

    async def evaluate_paper(
        self,
        paper_metadata: PaperMetadata,
        human_annotation: HumanAnnotation,
        modes: List[ExtractionPrompt],
        evaluation_metrics: Optional[List[EvaluationMetric]] = None,
        judge_prompt_template: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate model performance against human annotations for a paper."""
        logger.info(
            f"Starting evaluation for paper: {paper_metadata.paper_id}",
        )

        # Default to all metrics if none specified
        if evaluation_metrics is None:
            evaluation_metrics = [EvaluationMetric.ALL]

        # Determine which metrics to use
        use_similarity = any(
            metric
            in [
                EvaluationMetric.COSINE_SIMILARITY,
                EvaluationMetric.BOTH,
                EvaluationMetric.ALL,
            ]
            for metric in evaluation_metrics
        )
        use_judge = any(
            metric
            in [
                EvaluationMetric.LLM_JUDGE,
                EvaluationMetric.BOTH,
                EvaluationMetric.ALL,
            ]
            for metric in evaluation_metrics
        )
        use_nli = any(
            metric in [EvaluationMetric.NLI, EvaluationMetric.ALL]
            for metric in evaluation_metrics
        )

        # Extract definitions for all requested modes
        try:
            extraction_results = (
                await self.definition_extractor.extract_definitions(
                    paper_metadata,
                    modes,
                )
            )
        except Exception as e:
            logger.error(
                f"Definition extraction failed for {paper_metadata.paper_id}: {e}",
            )
            extraction_results = []

        # Initialize result containers
        similarity_results: List[SimilarityResult] = []
        judge_results: List[JudgeResult] = []
        nli_results: List[NLIResult] = []

        # Process each combination of human concept/definition and model extraction
        for i, human_concept in enumerate(human_annotation.defined_concepts):
            if i >= len(human_annotation.definitions):
                continue

            human_definition = human_annotation.definitions[i]

            for extraction_result in extraction_results:
                if not extraction_result.is_successful:
                    continue

                for model_definition_obj in extraction_result.definitions:
                    # Cosine similarity evaluation
                    if use_similarity and self.embedding_client:
                        similarity_result = await self._evaluate_similarity(
                            paper_metadata.paper_id,
                            human_concept,
                            human_definition,
                            extraction_result.mode,
                            model_definition_obj.definition_text,
                        )
                        similarity_results.append(similarity_result)

                    # LLM judge evaluation
                    if use_judge and self.llm_judge:
                        judge_result = await self._evaluate_judge(
                            paper_metadata.paper_id,
                            human_concept,
                            human_definition,
                            extraction_result.mode,
                            model_definition_obj.definition_text,
                            judge_prompt_template,
                        )
                        judge_results.append(judge_result)

                    # NLI evaluation
                    if use_nli and self.nli_client:
                        nli_result = await self._evaluate_nli(
                            paper_metadata.paper_id,
                            human_concept,
                            human_definition,
                            extraction_result.mode,
                            model_definition_obj.definition_text,
                        )
                        nli_results.append(nli_result)

        result = EvaluationResult(
            paper_id=paper_metadata.paper_id,
            human_annotations=human_annotation,
            extraction_results=extraction_results,
            similarity_results=similarity_results,
            judge_results=judge_results,
            nli_results=nli_results,
        )

        logger.info(
            f"Evaluation complete for {paper_metadata.paper_id}: "
            f"{len(similarity_results)} similarity, {len(judge_results)} judge, {len(nli_results)} NLI results",
        )

        return result

    async def _evaluate_similarity(
        self,
        paper_id: str,
        human_concept: str,
        human_definition: str,
        model_mode: ExtractionPrompt,
        model_definition: str,
    ) -> SimilarityResult:
        """Evaluate similarity using embeddings."""
        try:
            if self.embedding_client is None:
                raise EmbeddingError("Embedding client is not configured")
            # Generate embeddings
            (
                human_embedding,
                human_error,
            ) = self.embedding_client.generate_embedding(
                human_definition,
            )
            if human_error:
                raise EmbeddingError(f"Human embedding failed: {human_error}")

            (
                model_embedding,
                model_error,
            ) = self.embedding_client.generate_embedding(
                model_definition,
            )
            if model_error:
                raise EmbeddingError(f"Model embedding failed: {model_error}")

            # Calculate similarity
            similarity_score = self.embedding_client.calculate_similarity(
                human_embedding,
                model_embedding,
            )

            return SimilarityResult(
                paper_id=paper_id,
                human_concept=human_concept,
                human_definition=human_definition,
                model_mode=model_mode,
                model_definition=model_definition,
                similarity_score=similarity_score,
            )

        except Exception as e:
            error_msg = f"Similarity evaluation failed: {e}"
            logger.error(error_msg)
            return SimilarityResult(
                paper_id=paper_id,
                human_concept=human_concept,
                human_definition=human_definition,
                model_mode=model_mode,
                model_definition=model_definition,
                embedding_error=error_msg,
            )

    async def _evaluate_judge(
        self,
        paper_id: str,
        human_concept: str,
        human_definition: str,
        model_mode: ExtractionPrompt,
        model_definition: str,
        prompt_template: Optional[str],
    ) -> JudgeResult:
        """Evaluate using LLM judge."""
        try:
            if self.llm_judge is None:
                raise ValueError("LLM judge is not configured")
            (
                judgment_category,
                raw_response,
                error,
            ) = await self.llm_judge.judge_definition_pair(
                human_definition,
                model_definition,
                human_concept,
                prompt_template,
            )

            return JudgeResult(
                paper_id=paper_id,
                human_concept=human_concept,
                human_definition=human_definition,
                model_mode=model_mode,
                model_definition=model_definition,
                judgment_category=judgment_category,
                judge_prompt_template=prompt_template,
                judgment_error=error,
            )

        except Exception as e:
            error_msg = f"Judge evaluation failed: {e}"
            logger.error(error_msg)
            return JudgeResult(
                paper_id=paper_id,
                human_concept=human_concept,
                human_definition=human_definition,
                model_mode=model_mode,
                model_definition=model_definition,
                judgment_error=error_msg,
            )

    async def _evaluate_nli(
        self,
        paper_id: str,
        human_concept: str,
        human_definition: str,
        model_mode: ExtractionPrompt,
        model_definition: str,
    ) -> NLIResult:
        """Evaluate using NLI."""
        try:
            nli_client = self.nli_client
            if nli_client is None:
                raise ValueError("NLI client is not configured")
            nli_client = cast(NLIClient, nli_client)
            return nli_client.evaluate_bidirectional_entailment(
                paper_id,
                human_concept,
                human_definition,
                model_mode,
                model_definition,
            )

        except Exception as e:
            error_msg = f"NLI evaluation failed: {e}"
            logger.error(error_msg)
            return NLIResult(
                paper_id=paper_id,
                human_concept=human_concept,
                human_definition=human_definition,
                model_mode=model_mode,
                model_definition=model_definition,
                nli_error=error_msg,
            )

    def calculate_aggregate_metrics(
        self,
        evaluation_results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """Calculate aggregate metrics across multiple papers."""
        all_similarities = []
        all_judgments = []
        all_nli_results = []

        for result in evaluation_results:
            all_similarities.extend(result.valid_similarities)
            all_judgments.extend(result.valid_judgments)
            all_nli_results.extend(result.valid_nli_results)

        metrics = {}

        # Similarity metrics
        if all_similarities:
            similarity_scores = [
                r.similarity_score
                for r in all_similarities
                if r.similarity_score is not None
            ]
            if similarity_scores:
                metrics["mean_similarity"] = float(np.mean(similarity_scores))
                metrics["std_similarity"] = float(np.std(similarity_scores))
                metrics["similarity_count"] = len(similarity_scores)

        # Judge metrics
        if all_judgments:
            same_count = sum(
                1
                for r in all_judgments
                if r.judgment_category == JudgeCategory.SAME.value
            )
            metrics["judge_same_percentage"] = (
                same_count / len(all_judgments)
            ) * 100
            metrics["judge_count"] = len(all_judgments)

        # NLI metrics
        if all_nli_results:
            entailment_scores = [
                r.entailment_score
                for r in all_nli_results
                if r.entailment_score is not None
            ]
            if entailment_scores:
                metrics["mean_entailment"] = float(np.mean(entailment_scores))
                metrics["nli_count"] = len(entailment_scores)

        return metrics
