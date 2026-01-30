import argparse
import asyncio

from _bootstrap import load_config_module
from scidef.benchmark import (
    estimate_token_use,
    load_msr_paraphrases,
    load_quora_duplicates,
    load_sick,
    load_sts3k,
    load_stsb,
)
from scidef.benchmark.judge_report import (
    JudgeBenchmarkResult,
    save_judge_tuning_results,
)
from scidef.benchmark.metrics import MeasureMethod, measure_performance
from scidef.benchmark.service import (
    _map_score_to_bucket,
    evaluate_llm_judge,
)
from scidef.model.dataclass import JudgeSystemPrompt
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


def parse_prompt_tuples(value):
    """Parse prompt tuple arguments in format 'system_prompt,judge_prompt'."""
    try:
        if value.startswith("(") and value.endswith(")"):
            # Remove parentheses and split by comma
            inner = value[1:-1]
            parts = [p.strip() for p in inner.split(",")]
            parts = [parts[0], ",".join(parts[1:])]
            if len(parts) != 2:
                raise ValueError("Expected exactly 2 values")
            system_prompt = JudgeSystemPrompt(parts[0])
            treshold = parse_thresholds(parts[1])

            return (system_prompt, treshold)
        else:
            raise ValueError(
                "Format should be '(system_prompt, treshold)'",
            )
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Invalid prompt tuple format: {value}. Error: {e}",
        )


async def main():
    parser = argparse.ArgumentParser(
        description="Run LLM judge benchmark evaluation",
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
        "--prompt-combinations",
        nargs="+",
        type=parse_prompt_tuples,
        default=[
            (
                JudgeSystemPrompt.BINARY,
                [[0.94]],
            ),
            (
                JudgeSystemPrompt.TERNARY,
                [[0.8, 0.94]],
            ),
            (
                JudgeSystemPrompt.CATEGORICAL4,
                [[0.7, 0.8, 0.95]],
            ),
        ],
        help="Prompt combinations as tuples: '(system_prompt,judge_prompt)'",
    )

    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[0.7],
        help="Temperature values to evaluate",
    )

    parser.add_argument(
        "--top-p-values",
        nargs="+",
        type=float,
        default=[0.95],
        help="Top-p values to evaluate",
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
        "--split",
        choices=["train", "test", "validation"],
        default="train",
        help="Dataset split",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Sample size for testing",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum number of tokens to run per dataset",
    )
    parser.add_argument(
        "--per-pair-concurrency",
        type=int,
        default=12,
        help="Max concurrent LLM requests per evaluation (default: 12).",
    )

    args = parser.parse_args()
    config = Config()

    # Create judge client
    judge_client = config.create_judge_client()
    if not judge_client:
        logger.error(
            "Failed to create judge client. Check your configuration.",
        )
        return

    all_results = []

    print("Starting LLM judge tuning evaluation...")
    print(f"   - Model: {judge_client.model_name}")
    print(f"   - Datasets: {args.datasets}")
    print(f"   - Temperatures: {args.temperatures}")
    print(f"   - Top-p values: {args.top_p_values}")
    print(f"   - Metrics: {[metric.value for metric in args.metrics]}")
    print(f"   - Split: {args.split}")
    print(f"   - Sample size: {args.sample_size or 'all'}")
    print(
        f"   - Prompt combinations: {[(p[0].value, p[1]) for p in args.prompt_combinations]}",
    )
    print(
        f"   - Ground truth thresholds: {args.ground_truth_thresholds or 'None'}",
    )
    print()

    total_thresholds = sum(
        len(thresholds) for _, thresholds in args.prompt_combinations
    )
    total_evaluations = (
        len(args.datasets)
        * total_thresholds
        * len(args.temperatures)
        * len(args.top_p_values)
        * len(args.metrics)
        * len(args.ground_truth_thresholds)
    )
    current_evaluation = 0

    for dataset_idx, dataset_name in enumerate(args.datasets, 1):
        print(
            f"Loading dataset {dataset_idx}/{len(args.datasets)}: {dataset_name}",
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

        print(f"   Loaded {len(pairs)} pairs")

        estimated_tokens = estimate_token_use(pairs)
        print(
            f"{estimated_tokens} input tokens estimated for dataset {dataset_name}.",
        )

        if args.max_tokens is not None and estimated_tokens > args.max_tokens:
            logger.warning(
                f"Too many tokens found in dataset {dataset_name} ({estimated_tokens}. Stopping.)",
            )
            break

        for (
            system_prompt,
            thresholds,
        ) in args.prompt_combinations:
            for temperature in args.temperatures:
                for top_p in args.top_p_values:
                    base_result = await evaluate_llm_judge(
                        llm_judge=judge_client,
                        pairs=pairs,
                        dataset_name=dataset_name,
                        temperature=temperature,
                        top_p=top_p,
                        system_prompt=system_prompt,
                        per_pair_concurrency=args.per_pair_concurrency,
                    )

                    base_predictions = base_result.result or []
                    meta_predictions = (base_result.meta or {}).get(
                        "all_predictions",
                        [],
                    )
                    ground_truth = [
                        prediction.get("ground_truth")
                        for prediction in meta_predictions
                    ]

                    for metric in args.metrics:
                        for gt_threshold in args.ground_truth_thresholds:
                            for threshold in thresholds:
                                current_evaluation += 1
                                print(
                                    "   🔍 Evaluating [{}/{}]: + {} (T={}, p={}, threshold={}, ground truth={}, {})".format(
                                        current_evaluation,
                                        total_evaluations,
                                        system_prompt.value,
                                        temperature,
                                        top_p,
                                        threshold,
                                        gt_threshold,
                                        metric.value,
                                    ),
                                )

                                correlation = None
                                if (
                                    len(base_predictions) >= 2
                                    and len(ground_truth) >= 2
                                ):
                                    if (
                                        threshold is not None
                                        and metric is not MeasureMethod.PEARSON
                                    ):
                                        if isinstance(threshold, list):
                                            bucketed_predictions = [
                                                _map_score_to_bucket(
                                                    pred,
                                                    threshold,
                                                )
                                                for pred in base_predictions
                                            ]
                                        else:
                                            bucketed_predictions = [
                                                1 if pred > threshold else 0
                                                for pred in base_predictions
                                            ]
                                    else:
                                        bucketed_predictions = base_predictions

                                    effective_gt_threshold = (
                                        gt_threshold
                                        if gt_threshold is not None
                                        else threshold
                                    )

                                    if (
                                        effective_gt_threshold is not None
                                        and metric is not MeasureMethod.PEARSON
                                    ):
                                        if isinstance(
                                            effective_gt_threshold,
                                            list,
                                        ):
                                            bucketed_ground_truth = [
                                                _map_score_to_bucket(
                                                    gt,
                                                    effective_gt_threshold,
                                                )
                                                for gt in ground_truth
                                            ]
                                        else:
                                            bucketed_ground_truth = [
                                                1
                                                if gt > effective_gt_threshold
                                                else 0
                                                for gt in ground_truth
                                            ]
                                    else:
                                        bucketed_ground_truth = ground_truth

                                    correlation = measure_performance(
                                        pred=bucketed_predictions,
                                        gt=bucketed_ground_truth,
                                        method=metric,
                                    )

                                    tuning_result = JudgeBenchmarkResult(
                                        dataset=dataset_name,
                                        system_prompt=system_prompt,
                                        temperature=temperature,
                                        top_p=top_p,
                                        threshold=threshold,
                                        ground_truth_threshold=gt_threshold,
                                        metric=metric.value,
                                        correlation=correlation,
                                        sample_size=args.sample_size
                                        or base_result.total_count,
                                        split=args.split,
                                        success_rate=base_result.success_rate,
                                        error_count=base_result.error_count,
                                        total_count=base_result.total_count,
                                        model_name=judge_client.model_name,
                                    )
                                all_results.append(tuning_result)

                                correlation_str = (
                                    f"{correlation:.4f}"
                                    if correlation is not None
                                    else "N/A"
                                )
                                success_str = (
                                    f"{base_result.success_rate:.3f}"
                                    if base_result.success_rate is not None
                                    else "N/A"
                                )
                                print(
                                    f"      → Correlation: {correlation_str}, Success Rate: {success_str}",
                                )

            print()

        # Generate and save reports
    if all_results:
        save_judge_tuning_results(all_results)
        print("\nReports saved:")
        print("   - JUDGE_Benchmark_RESULTS.md (latest results)")
        print("   - results/judge_benchmark/ (detailed timestamped results)")

        # Show quick summary
        valid_results = [r for r in all_results if r.correlation is not None]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x.correlation)
            print("\n Best result:")
            print(f"   Dataset: {best_result.dataset}")
            print(f"   System Prompt: {best_result.system_prompt.value}")
            print(f"   Temperature: {best_result.temperature}")
            print(f"   Top-p: {best_result.top_p}")
            print(f"   threshold: {best_result.threshold}")
            print(
                f"   Ground truth threshold: {best_result.ground_truth_threshold}",
            )
            print(f"   Correlation: {best_result.correlation:.4f}")
            print(f"   Success Rate: {best_result.success_rate:.3f}")
    else:
        print(
            "No results generated - check your judge client configuration",
        )


if __name__ == "__main__":
    asyncio.run(main())
