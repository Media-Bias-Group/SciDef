import argparse
import asyncio

from config import Config
from scidef.benchmark import (
    load_msr_paraphrases,
    load_quora_duplicates,
    load_sick,
    load_sts3k,
    load_stsb,
)
from scidef.benchmark.metrics import MeasureMethod
from scidef.benchmark.nli_report import (
    NLIBenchmarkResult,
    save_nli_tuning_results,
)
from scidef.benchmark.service import evaluate_nli
from scidef.model.nli import NLIMode
from scidef.model.nli.dataclass import ScoreMode
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


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
    parser = argparse.ArgumentParser(
        description="Run NLI benchmark evaluation",
    )
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
        help="Datasets to evaluate (default: all datasets)",
    )

    parser.add_argument(
        "--score-modes",
        nargs="+",
        type=ScoreMode,
        choices=list(ScoreMode),
        default=[ScoreMode.HMEAN, ScoreMode.AMEAN],
        help="Score modes to evaluate for bidirectional NLI",
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=parse_thresholds,
        default=[0.5],
        help="Thresholds to evaluate (can include lists like '--threshold 0.6 0.7 0.8 0.9 0.95')",
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
        "--models",
        nargs="+",
        default=[
            "cross-encoder/nli-deberta-v3-large",
            "facebook/bart-large-mnli",
            "roberta-large-mnli",
            "tasksource/ModernBERT-large-nli",
            "dleemiller/ModernCE-large-nli",
            "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
            "microsoft/deberta-xlarge-mnli",
            "pritamdeka/PubMedBERT-MNLI-MedNLI",
        ],
        help="NLI models to evaluate (default: all three models)",
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

    all_results = []

    print(" Starting NLI benchmark evaluation...")
    print(f"   - Models: {args.models}")
    print(f"   - Datasets: {args.datasets}")
    print(f"   - Score modes: {[mode.value for mode in args.score_modes]}")
    print(f"   - Metrics: {[metric.value for metric in args.metrics]}")
    print(f"   - Split: {args.split}")
    print(f"   - Sample size: {args.sample_size or 'all'}")
    print(f"   - Thresholds: {args.thresholds or 'None'}")
    print(
        f"   - Ground truth thresholds: {args.ground_truth_thresholds or 'None'}",
    )
    print()

    total_evaluations = (
        len(args.models)
        * len(args.datasets)
        * len(args.score_modes)
        * len(args.metrics)
        * len(args.ground_truth_thresholds)
    )
    current_evaluation = 0

    for model_idx, model_name in enumerate(args.models, 1):
        print(
            f" Evaluating model {model_idx}/{len(args.models)}: {model_name}",
        )

        nli_client = config.create_nli_client(model_name)
        if not nli_client:
            logger.error(
                f"Failed to create NLI client for model: {model_name}. Skipping...",
            )
            continue

        for dataset_idx, dataset_name in enumerate(args.datasets, 1):
            print(
                f"    Loading dataset {dataset_idx}/{len(args.datasets)}: {dataset_name}",
            )

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

            if not pairs:
                logger.warning(f"No data loaded for dataset: {dataset_name}")
                continue

            print(f"      Loaded {len(pairs)} pairs")

            for gt_threshold in args.ground_truth_thresholds:
                print(f"       Ground truth threshold: {gt_threshold}")

                result = await evaluate_nli(
                    nli_client=nli_client,
                    pairs=pairs,
                    dataset_name=dataset_name,
                    mode=NLIMode.BIDIRECTIONAL,  # Always use bidirectional
                    score_mode=args.score_modes,
                    metric=args.metrics,
                    threshold=args.thresholds,
                    ground_truth_threshold=gt_threshold,
                )

                # Create NLI-specific tuning result
                for m, metric in enumerate(args.metrics):
                    for s, score_mode in enumerate(
                        [ScoreMode.HMEAN, ScoreMode.AMEAN],
                    ):
                        if score_mode not in args.score_modes:
                            continue
                        current_evaluation += 1
                        print(
                            f"       Evaluating [{current_evaluation}/{total_evaluations}]: {score_mode} with {metric.value}, threshold {args.thresholds}, ground truth {gt_threshold}",
                        )
                        assert result.correlations is not None

                        correlation = result.correlations[
                            m * len(args.score_modes) + s
                        ]
                        tuning_result = NLIBenchmarkResult(
                            dataset=dataset_name,
                            score_mode=score_mode,
                            metric=metric.value,
                            correlation=correlation,
                            sample_size=args.sample_size or len(pairs),
                            split=args.split,
                            success_rate=result.success_rate,
                            error_count=result.error_count,
                            total_count=result.total_count,
                            model_name=model_name,
                            threshold=args.thresholds,
                            ground_truth_threshold=gt_threshold,
                        )
                        all_results.append(tuning_result)

                    # Show immediate result
                    correlation_str = (
                        f"{correlation:.4f}"
                        if correlation is not None
                        else "N/A"
                    )
                    success_str = (
                        f"{result.success_rate:.3f}"
                        if result.success_rate is not None
                        else "N/A"
                    )
                    print(
                        f"         → Correlation: {correlation_str}, Success Rate: {success_str}",
                    )

            print()

        print(f"    Completed evaluation for {model_name}")
        print()

    # Generate and save reports
    if all_results:
        print("=" * 60)
        save_nli_tuning_results(all_results)

        print("\n Reports saved:")
        print("   - NLI_BENCHMARK_RESULTS.md (latest results)")
        print("   - results/nli_benchmark/ (detailed timestamped results)")

        # Show quick summary
        valid_results = [r for r in all_results if r.correlation is not None]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x.correlation)
            print("\n Best result:")
            print(f"   Dataset: {best_result.dataset}")
            print(f"   Score Mode: {best_result.score_mode.value}")
            print(f"   Metric: {best_result.metric}")
            print(f"   Correlation: {best_result.correlation:.4f}")
            print(f"   Success Rate: {best_result.success_rate:.3f}")
    else:
        print("No results generated - check your NLI client configuration")


if __name__ == "__main__":
    asyncio.run(main())
