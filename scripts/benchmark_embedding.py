import argparse
import asyncio

from _bootstrap import load_config_module
from scidef.benchmark import (
    load_msr_paraphrases,
    load_quora_duplicates,
    load_sick,
    load_sts3k,
    load_stsb,
)
from scidef.benchmark.embedding_report import (
    EmbeddingBenchmarkResult,
    save_embedding_tuning_results,
)
from scidef.benchmark.metrics import MeasureMethod, measure_performance
from scidef.benchmark.service import (
    _map_score_to_bucket,
    evaluate_cosine_similarity,
)
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)
Config = load_config_module().Config


def parse_thresholds(value):
    """Parse threshold arguments that can be either floats or lists of floats."""
    try:
        return float(value)
    except ValueError:
        if value.startswith("[") and value.endswith("]"):
            list_str = value[1:-1]
            return [float(x.strip()) for x in list_str.split(",")]
        else:
            raise argparse.ArgumentTypeError(
                f"Invalid threshold format: {value}",
            )


async def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "stsb",
            "sick",
            "sts3k-all",
            "sts3k-non",
            "sts3k-adv",
            "msr-paraphrases",
            "quora-duplicates",
        ],
        default=[
            "stsb",
            "sick",
            "sts3k-all",
            "sts3k-non",
            "sts3k-adv",
            "msr-paraphrases",
            "quora-duplicates",
        ],
        help="Datasets to evaluate",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=parse_thresholds,
        default=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, [0.6, 0.7, 0.8, 0.9]],
        help="Thresholds to evaluate (can include lists like [0.6,0.7,0.8,0.9])",
    )

    parser.add_argument(
        "--ground-truth-thresholds",
        nargs="+",
        type=parse_thresholds,
        default=[0.8, 0.85, 0.9, 0.95],
        help=(
            "Ground truth thresholds to evaluate (applied to labels; accepts floats or "
            "lists such as '[0.6,0.7,0.8,0.9]')"
        ),
    )

    parser.add_argument(
        "--metrics",
        nargs="+",
        type=MeasureMethod,
        choices=list(MeasureMethod),
        default=list(MeasureMethod),
        help="Metrics to evaluate",
    )

    parser.add_argument(
        "--split",
        choices=["train", "test", "validation"],
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Sample size for testing",
    )
    args = parser.parse_args()
    config = Config()

    embedding_client = config.create_embedding_client()
    if not embedding_client:
        logger.warning(
            "No embedding client available for cosine similarity. Need EMBEDDING_PROVIDER and related config.",
        )

    all_results = []

    print("Starting embedding tuning...")
    print(f"   - Datasets: {args.datasets}")
    print(f"   - Thresholds: {args.thresholds}")
    print(
        f"   - Ground truth thresholds: {args.ground_truth_thresholds or 'None'}",
    )
    print()

    for dataset_name in args.datasets:
        if dataset_name == "stsb":
            pairs = load_stsb(args.split, args.sample_size)
        elif dataset_name == "sick":
            pairs = load_sick(args.split, args.sample_size)
        elif "sts3k" in dataset_name:
            pairs = load_sts3k(dataset_name, args.sample_size)
        elif dataset_name == "msr-paraphrases":
            pairs = load_msr_paraphrases(args.split, args.sample_size)
        elif dataset_name == "quora-duplicates":
            pairs = load_quora_duplicates(args.split, args.sample_size)
        else:
            logger.warning(f"Unknown dataset: {dataset_name}")
            continue

        result = await evaluate_cosine_similarity(
            embedding_client,
            pairs,
            dataset_name,
        )

        base_predictions = result.result or []
        meta_predictions = (result.meta or {}).get("all_predictions", [])
        ground_truth = [
            prediction.get("ground_truth") for prediction in meta_predictions
        ]

        for metric in args.metrics:
            for gt_threshold in args.ground_truth_thresholds:
                for threshold in args.thresholds:
                    print(
                        "data: {}, threshold: {}, ground truth: {}, metric: {}".format(
                            dataset_name,
                            threshold,
                            gt_threshold,
                            metric,
                        ),
                    )
                    correlation = None
                    if len(base_predictions) >= 2 and len(ground_truth) >= 2:
                        effective_gt_threshold = (
                            gt_threshold
                            if gt_threshold is not None
                            else threshold
                        )

                        if threshold and metric is not MeasureMethod.PEARSON:
                            if isinstance(threshold, list):
                                bucketed_predictions = [
                                    _map_score_to_bucket(pred, threshold)
                                    for pred in base_predictions
                                ]
                            else:
                                bucketed_predictions = [
                                    1 if pred > threshold else 0
                                    for pred in base_predictions
                                ]
                        else:
                            bucketed_predictions = base_predictions

                        if (
                            effective_gt_threshold is not None
                            and metric is not MeasureMethod.PEARSON
                        ):
                            if isinstance(effective_gt_threshold, list):
                                bucketed_ground_truth = [
                                    _map_score_to_bucket(
                                        gt,
                                        effective_gt_threshold,
                                    )
                                    for gt in ground_truth
                                ]
                            else:
                                bucketed_ground_truth = [
                                    1 if gt > effective_gt_threshold else 0
                                    for gt in ground_truth
                                ]
                        else:
                            bucketed_ground_truth = ground_truth

                        correlation = measure_performance(
                            pred=bucketed_predictions,
                            gt=bucketed_ground_truth,
                            method=metric,
                        )

                        tuning_result = EmbeddingBenchmarkResult(
                            dataset=dataset_name,
                            threshold=threshold,
                            metric=str(metric),
                            correlation=correlation,
                            sample_size=args.sample_size or "all",
                            split=args.split,
                            ground_truth_threshold=gt_threshold,
                        )
                        all_results.append(tuning_result)

    if all_results:
        save_embedding_tuning_results(all_results)
        print("\n Reports saved:")
        print("  - EMBEDDING_BENCHMARK_RESULTS.md (latest results)")
        print(
            "  - results/embedding_benchmark/ (detailed timestamped results)",
        )


if __name__ == "__main__":
    asyncio.run(main())
