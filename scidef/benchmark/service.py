import asyncio
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from scidef.benchmark import (
    BenchmarkResult,
    load_sick,
    load_sts3k,
    load_stsb,
)
from scidef.benchmark.metrics import MeasureMethod, measure_performance
from scidef.model.dataclass import ExtractionPrompt, JudgeSystemPrompt
from scidef.model.embedding.client import EmbeddingClient
from scidef.model.llm.judge.client import JudgeClient
from scidef.model.nli import NLIClient, NLIMode
from scidef.model.nli.dataclass import ScoreMode
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


async def run_benchmarks(
    embedding_client: Optional[EmbeddingClient] = None,
    llm_judge: Optional[JudgeClient] = None,
    nli_client: Optional[NLIClient] = None,
    datasets: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    split: str = "train",
    sample_size: Optional[int] = None,
    judge_temperature: Optional[float] = None,
    judge_top_p: Optional[int] = None,
    judge_system_prompt: Optional[JudgeSystemPrompt] = None,
) -> List[BenchmarkResult]:
    """Run simple benchmark evaluation."""
    if datasets is None:
        datasets = ["stsb", "sick", "sts3k"]
    if metrics is None:
        metrics = ["cosine_similarity"]

    results: List[BenchmarkResult] = []

    for dataset_name in datasets:
        if dataset_name == "stsb":
            pairs = load_stsb(split, sample_size)
        elif dataset_name == "sick":
            pairs = load_sick(split, sample_size)
        elif dataset_name == "sts3k":
            pairs = load_sts3k(split, sample_size)
        else:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue

        if not pairs:
            continue

        for metric in metrics:
            logger.info(
                f"Evaluating {metric} on {dataset_name} ({len(pairs)} pairs)",
            )

            if metric == "cosine_similarity":
                result = await evaluate_cosine_similarity(
                    embedding_client,
                    pairs,
                    dataset_name,
                )
                results.append(result)
            elif metric == "llm_judge":
                result = await evaluate_llm_judge(
                    llm_judge=llm_judge,
                    pairs=pairs,
                    dataset_name=dataset_name,
                    temperature=judge_temperature,
                    top_p=judge_top_p,
                    system_prompt=judge_system_prompt,
                )
                results.append(result)
            elif metric == "nli":
                result = await evaluate_nli(nli_client, pairs, dataset_name)
                results.append(result)
            else:
                logger.warning(f"Unknown metric: {metric}")
                continue

    return results


async def evaluate_llm_judge(
    llm_judge: Optional[JudgeClient],
    pairs: List[Tuple[str, str, float]],
    dataset_name: str,
    threshold: Optional[Union[List[float], float]] = None,
    ground_truth_threshold: Optional[Union[List[float], float]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[int] = None,
    system_prompt: Optional[JudgeSystemPrompt] = None,
    metric: MeasureMethod = MeasureMethod.PEARSON,
    per_pair_concurrency: int = 12,
) -> BenchmarkResult:
    """Evaluate LLM judge on sentence pairs (concurrent per-pair)."""
    if not llm_judge:
        logger.error(f"No LLM judge available for {dataset_name}")
        return BenchmarkResult(
            metric="llm_judge",
            result=[],
            dataset=dataset_name,
            error_count=len(pairs),
            total_count=len(pairs),
            model_name="N/A",
        )

    n = len(pairs)
    predictions: List[Optional[float]] = [None] * n
    ground_truth: List[Optional[float]] = [None] * n
    all_predictions: List[Dict] = []
    errors = 0

    sem = asyncio.Semaphore(per_pair_concurrency)

    async def one(i: int, sent1: str, sent2: str, score: float):
        async with sem:
            try:
                result = await llm_judge.judge_definition_pair(
                    human_definition=sent1,
                    model_definition=sent2,
                    temperature=temperature,
                    top_p=top_p,
                    system_prompt_type=system_prompt,
                )
                if not result or result[0] is None:
                    return i, None, score, "empty"
                return i, float(result[0]), score, None
            except Exception as e:
                return i, None, score, str(e)

    tasks = [
        asyncio.create_task(one(i, s1, s2, score))
        for i, (s1, s2, score) in enumerate(pairs)
    ]

    # Stream completions as they finish
    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        i, pred, gt, err = await fut
        if err is not None:
            logger.warning(
                f"LLM judge error for pair {i + 1} in {dataset_name}: {err}",
            )
            errors += 1
            continue
        predictions[i] = pred
        ground_truth[i] = gt
        all_predictions.append({"prediction": pred, "ground_truth": gt})

    # Filter out any failed slots (keep alignment)
    filtered_preds: List[float] = []
    filtered_gt: List[float] = []
    for p, g in zip(predictions, ground_truth):
        if p is not None and g is not None:
            filtered_preds.append(p)
            filtered_gt.append(g)

    correlation = None
    if len(filtered_preds) >= 2 and len(filtered_gt) >= 2:
        effective_gt_threshold = (
            ground_truth_threshold
            if ground_truth_threshold is not None
            else threshold
        )

        if threshold is not None and metric is not MeasureMethod.PEARSON:
            if isinstance(threshold, list):
                filtered_preds = [
                    _map_score_to_bucket(pred, threshold)
                    for pred in filtered_preds
                ]
            else:
                filtered_preds = [
                    1 if pred > threshold else 0 for pred in filtered_preds
                ]

        if (
            effective_gt_threshold is not None
            and metric is not MeasureMethod.PEARSON
        ):
            if isinstance(effective_gt_threshold, list):
                filtered_gt = [
                    _map_score_to_bucket(gt, effective_gt_threshold)
                    for gt in filtered_gt
                ]
            else:
                filtered_gt = [
                    1 if gt > effective_gt_threshold else 0
                    for gt in filtered_gt
                ]

        correlation = measure_performance(
            pred=filtered_preds,
            gt=filtered_gt,
            method=metric,
        )

    return BenchmarkResult(
        metric="llm_judge",
        result=filtered_preds,
        dataset=dataset_name,
        correlation=correlation,
        success_rate=(len(filtered_preds)) / n if n else 0.0,
        error_count=errors,
        total_count=n,
        model_name=llm_judge.model_name,
        meta={"all_predictions": all_predictions},
    )


async def evaluate_nli(
    nli_client: Optional[NLIClient],
    pairs: List[Tuple[str, str, float]],
    dataset_name: str,
    mode: NLIMode = NLIMode.BIDIRECTIONAL,
    score_mode: Union[List[ScoreMode], ScoreMode] = ScoreMode.HMEAN,
    metric: Union[List[MeasureMethod], MeasureMethod] = MeasureMethod.PEARSON,
    threshold: Optional[Union[float, List[float]]] = None,
    ground_truth_threshold: Optional[Union[float, List[float]]] = None,
) -> BenchmarkResult:
    """Evaluate NLI on sentence pairs."""
    if not nli_client:
        logger.warning(f"No NLI client available for {dataset_name}")
        return BenchmarkResult(
            metric="nli",
            result=[],
            dataset=dataset_name,
            error_count=len(pairs),
            total_count=len(pairs),
            model_name="N/A",
        )

    predictions: List[Optional[List[float] | float]] = []
    all_predictions: List[Dict] = []
    ground_truth: List[float] = []
    errors = 0

    if not isinstance(metric, list):
        metric = [metric]
    if not isinstance(score_mode, list):
        score_mode = [score_mode]

    for i, (sent1, sent2, score) in enumerate(pairs):
        try:
            # Use NLI client's bidirectional entailment for similarity
            nli_result = await nli_client.run(
                paper_id="benchmark",
                human_concept="",
                human_definition=sent1,
                model_mode=ExtractionPrompt.EXTRACTIVE,
                model_definition=sent2,
                mode=mode,
                score_mode=score_mode,
            )
            if nli_result and nli_result.entailment_score is not None:
                if mode == NLIMode.BIDIRECTIONAL:
                    predictions.append(nli_result.bidirectional_scores)
                    ground_truth.append(score)
                else:
                    predictions.append(nli_result.entailment_score)
                    ground_truth.append(score)

                all_predictions.append(
                    {
                        "prediction": nli_result.predicted_label,
                        "forward_entailment_score": nli_result.forward_entailment_score,
                        "backward_entailment_score": nli_result.backward_entailment_score,
                        "forward_predicted_label": nli_result.forward_predicted_label,
                        "backward_predicted_label": nli_result.backward_predicted_label,
                        "ground_truth": score,
                    },
                )

            else:
                logger.warning(
                    f"NLI returned no entailment score for pair {i + 1} in {dataset_name}",
                )
                errors += 1

        except Exception as e:
            logger.warning(
                f"NLI error for pair {i + 1} in {dataset_name}: {e}",
            )
            errors += 1

    correlations = []
    if len(predictions) >= 2 and len(ground_truth) >= 2:
        for m in metric:  # For each measure
            for s, smode in enumerate(
                [ScoreMode.HMEAN, ScoreMode.AMEAN],
            ):  # For each score mode
                if smode not in score_mode:
                    continue

                bucketed_predictions = np.asarray(predictions)[:, s]
                bucketed_ground_truth = ground_truth

                if threshold and m is not MeasureMethod.PEARSON:
                    if isinstance(threshold, list):
                        bucketed_predictions = [
                            float(_map_score_to_bucket(pred, threshold))  # type: ignore
                            for pred in bucketed_predictions
                        ]
                    else:
                        bucketed_predictions = [
                            1 if pred > threshold else 0
                            for pred in bucketed_predictions
                        ]

                effective_gt_threshold = (
                    ground_truth_threshold
                    if ground_truth_threshold is not None
                    else threshold
                )

                if (
                    effective_gt_threshold is not None
                    and m is not MeasureMethod.PEARSON
                ):
                    if isinstance(effective_gt_threshold, list):
                        bucketed_ground_truth = [
                            _map_score_to_bucket(gt, effective_gt_threshold)
                            for gt in bucketed_ground_truth
                        ]
                    else:
                        bucketed_ground_truth = [
                            1 if gt > effective_gt_threshold else 0
                            for gt in bucketed_ground_truth
                        ]

                correlations.append(
                    measure_performance(
                        pred=bucketed_predictions,
                        gt=bucketed_ground_truth,
                        method=m,
                    ),
                )

    return BenchmarkResult(
        metric="nli",
        result=predictions,
        dataset=dataset_name,
        correlations=correlations,
        success_rate=(len(pairs) - errors) / len(pairs),
        error_count=errors,
        total_count=len(pairs),
        model_name=getattr(nli_client, "model_name", "Unknown"),
        meta={
            "all_predictions": all_predictions,
            "labels": nli_client.id2label,
        },
    )


def _map_score_to_bucket(score: float, thresholds: List[float]) -> int:
    """Map a score to a bucket based on thresholds."""
    for i, threshold in enumerate(thresholds):
        if score < threshold:
            return i
    return len(thresholds)


async def evaluate_cosine_similarity(
    embedding_client: Optional[EmbeddingClient],
    pairs: List[Tuple[str, str, float]],
    dataset_name: str,
    metric: MeasureMethod = MeasureMethod.PEARSON,
    threshold: Optional[Union[float, List[float]]] = None,
    ground_truth_threshold: Optional[Union[float, List[float]]] = None,
) -> BenchmarkResult:
    """Evaluate cosine similarity on sentence pairs using embeddings."""
    if not embedding_client:
        logger.warning(f"No embedding client available for {dataset_name}")
        return BenchmarkResult(
            metric="cosine_similarity",
            result=[],
            dataset=dataset_name,
            error_count=len(pairs),
            total_count=len(pairs),
            model_name="N/A",
        )

    predictions: List[float] = []
    all_predictions: List[Dict] = []
    ground_truth: List[float] = []
    errors = 0

    for i, (sent1, sent2, score) in enumerate(pairs):
        try:
            emb1, err1 = embedding_client.generate_embedding(sent1)
            emb2, err2 = embedding_client.generate_embedding(sent2)

            if err1 or err2 or not emb1 or not emb2:
                if err1:
                    logger.warning(
                        f"Embedding error for pair {i + 1} in {dataset_name}: {err1}",
                    )
                if err2:
                    logger.warning(
                        f"Embedding error for pair {i + 1} in {dataset_name}: {err2}",
                    )
                errors += 1
                continue

            similarity = embedding_client.calculate_similarity(emb1, emb2)
            predictions.append(similarity)
            ground_truth.append(score)
            all_predictions.append(
                {
                    "prediction": similarity,
                    "ground_truth": score,
                },
            )

        except Exception as e:
            logger.warning(
                f"Cosine similarity error for pair {i + 1} in {dataset_name}: {e}",
            )
            errors += 1

    correlation = None
    if len(predictions) >= 2 and len(ground_truth) >= 2:
        effective_gt_threshold = (
            ground_truth_threshold
            if ground_truth_threshold is not None
            else threshold
        )

        processed_predictions = predictions
        processed_ground_truth = ground_truth

        if threshold and metric is not MeasureMethod.PEARSON:
            if isinstance(threshold, list):
                processed_predictions = [
                    _map_score_to_bucket(pred, threshold)
                    for pred in processed_predictions
                ]
            else:
                processed_predictions = [
                    1 if pred > threshold else 0
                    for pred in processed_predictions
                ]

        if (
            effective_gt_threshold is not None
            and metric is not MeasureMethod.PEARSON
        ):
            if isinstance(effective_gt_threshold, list):
                processed_ground_truth = [
                    _map_score_to_bucket(gt, effective_gt_threshold)
                    for gt in processed_ground_truth
                ]
            else:
                processed_ground_truth = [
                    1 if gt > effective_gt_threshold else 0
                    for gt in processed_ground_truth
                ]

        correlation = measure_performance(
            pred=processed_predictions,
            gt=processed_ground_truth,
            method=metric,
        )

    return BenchmarkResult(
        metric="cosine_similarity",
        result=predictions,
        dataset=dataset_name,
        correlation=correlation,
        success_rate=(len(pairs) - errors) / len(pairs),
        error_count=errors,
        total_count=len(pairs),
        model_name=getattr(embedding_client, "model_name", "Unknown"),
        meta={
            "all_predictions": all_predictions,
            "method": metric,
            "threshold": threshold,
        },
    )
