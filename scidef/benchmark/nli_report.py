from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from scidef.model.nli.dataclass import ScoreMode
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


@dataclass
class NLIBenchmarkResult:
    """Result from NLI benchmark evaluation."""

    dataset: str
    score_mode: ScoreMode
    metric: str = field(default="")
    correlation: Optional[float] = None
    sample_size: int = 0
    split: str = "train"
    success_rate: Optional[float] = None
    error_count: int = 0
    total_count: int = 0
    model_name: str = "Unknown"
    threshold: Union[float, List[float]] = 0.0
    ground_truth_threshold: Optional[Union[float, List[float]]] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not self.metric:
            raise ValueError("metric must be provided for NLIBenchmarkResult")
        if self.timestamp is None:
            self.timestamp = datetime.now()


def generate_nli_report(results: List[NLIBenchmarkResult]) -> str:
    """Generate a comprehensive markdown report for NLI benchmark results."""
    if not results:
        return "# NLI Benchmark Results\n\nNo results available.\n"

    markdown = "# NLI Benchmark Results\n\n"
    markdown += (
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    )
    markdown += "**Evaluation Mode**: Bidirectional NLI\n\n"

    models = list(
        set(r.model_name for r in results if r.model_name != "Unknown"),
    )
    if models:
        markdown += f"**Models Evaluated**: {', '.join(models)}\n\n"

    valid_results = [r for r in results if r.correlation is not None]
    if valid_results:
        avg_correlation = sum(
            r.correlation for r in valid_results if r.correlation is not None
        ) / len(
            valid_results,
        )
        best_correlation = max(
            r.correlation for r in valid_results if r.correlation is not None
        )
        markdown += "## Summary Statistics\n\n"
        markdown += f"- **Total Evaluations**: {len(results)}\n"
        markdown += f"- **Successful Evaluations**: {len(valid_results)}\n"
        markdown += f"- **Average Correlation**: {avg_correlation:.4f}\n"
        markdown += f"- **Best Correlation**: {best_correlation:.4f}\n\n"

    markdown += "## Detailed Results\n\n"
    markdown += "| Dataset | Score Mode | Metric | Threshold | GT Threshold | Correlation | Success Rate | Sample Size | Split | Model |\n"
    markdown += "|---------|------------|--------|-----------|--------------|-------------|--------------|-------------|-------|-------|\n"

    sorted_results = sorted(
        results,
        key=lambda x: (
            x.correlation if x.correlation is not None else -1,
            x.dataset,
        ),
        reverse=True,
    )

    for result in sorted_results:
        correlation_str = (
            f"{result.correlation:.4f}"
            if result.correlation is not None
            else "N/A"
        )
        success_rate_str = (
            f"{result.success_rate:.3f}"
            if result.success_rate is not None
            else "N/A"
        )
        markdown += (
            f"| {result.dataset} | {result.score_mode.value} | {result.metric} | {result.threshold} | "
            f"{result.ground_truth_threshold} | {correlation_str} | {success_rate_str} | {result.sample_size} | "
            f"{result.split} | {result.model_name} |\n"
        )

    markdown += "\n## Best Configurations by Dataset\n\n"
    datasets = list(set(r.dataset for r in results))

    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r.dataset == dataset]
        valid_results = [
            r for r in dataset_results if r.correlation is not None
        ]

        if not valid_results:
            continue

        best_result = max(valid_results, key=lambda x: x.correlation)
        markdown += f"### {dataset.upper()} Dataset\n"
        markdown += f"- **Best Score Mode**: {best_result.score_mode.value}\n"
        markdown += f"- **Metric**: {best_result.metric}\n"
        markdown += f"- **Correlation**: {best_result.correlation:.4f}\n"
        markdown += (
            f"- **Success Rate**: {best_result.success_rate:.3f}\n"
            if best_result.success_rate
            else ""
        )
        markdown += f"- **Ground Truth Threshold**: {best_result.ground_truth_threshold}\n"
        markdown += f"- **Sample Size**: {best_result.sample_size}\n"
        markdown += f"- **Model**: {best_result.model_name}\n\n"

    # Model comparison
    markdown += "## Model Performance Analysis\n\n"
    models = list(
        set(r.model_name for r in results if r.model_name != "Unknown"),
    )

    if len(models) > 1:
        for model in sorted(models):
            model_results = [
                r
                for r in results
                if r.model_name == model and r.correlation is not None
            ]

            if model_results:
                avg_corr = sum(
                    r.correlation
                    for r in model_results
                    if r.correlation is not None
                ) / len(
                    model_results,
                )
                max_corr = max(
                    r.correlation
                    for r in model_results
                    if r.correlation is not None
                )
                min_corr = min(
                    r.correlation
                    for r in model_results
                    if r.correlation is not None
                )
                markdown += f"### {model}\n"
                markdown += f"- **Average Correlation**: {avg_corr:.4f}\n"
                markdown += f"- **Best Correlation**: {max_corr:.4f}\n"
                markdown += f"- **Worst Correlation**: {min_corr:.4f}\n"
                markdown += f"- **Evaluations**: {len(model_results)}\n\n"

    markdown += "## Score Mode Performance Analysis\n\n"
    score_modes = list(set(r.score_mode for r in results))

    for score_mode in sorted(score_modes, key=lambda x: x.value):
        mode_results = [
            r
            for r in results
            if r.score_mode == score_mode and r.correlation is not None
        ]

        if mode_results:
            avg_corr = sum(
                r.correlation
                for r in mode_results
                if r.correlation is not None
            ) / len(
                mode_results,
            )
            max_corr = max(
                r.correlation
                for r in mode_results
                if r.correlation is not None
            )
            min_corr = min(
                r.correlation
                for r in mode_results
                if r.correlation is not None
            )
            markdown += f"### {score_mode.value}\n"
            markdown += f"- **Average Correlation**: {avg_corr:.4f}\n"
            markdown += f"- **Best Correlation**: {max_corr:.4f}\n"
            markdown += f"- **Worst Correlation**: {min_corr:.4f}\n"
            markdown += f"- **Evaluations**: {len(mode_results)}\n\n"

    markdown += "## Performance Insights\n\n"

    models = list(
        set(r.model_name for r in valid_results if r.model_name != "Unknown"),
    )
    if len(models) > 1:
        model_avg_scores = {}
        for model in models:
            model_results = [r for r in valid_results if r.model_name == model]
            if model_results:
                model_avg_scores[model] = sum(
                    r.correlation
                    for r in model_results
                    if r.correlation is not None
                ) / len(model_results)

        if model_avg_scores:
            best_model = max(
                model_avg_scores,
                key=lambda x: model_avg_scores.get(x),
            )
            worst_model = min(
                model_avg_scores,
                key=lambda x: model_avg_scores.get(x),
            )

            markdown += "### Model Rankings\n"
            for i, (model, avg_score) in enumerate(
                sorted(
                    model_avg_scores.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ),
                1,
            ):
                markdown += f"{i}. **{model}**: {avg_score:.4f}\n"

            markdown += f"\n- **Best performing model**: {best_model} ({model_avg_scores[best_model]:.4f})\n"
            if len(models) > 1:
                improvement = (
                    model_avg_scores[best_model]
                    - model_avg_scores[worst_model]
                )
                markdown += f"- **Performance gap**: {improvement:.4f} between best and worst\n"

        markdown += "\n### Top Model-Dataset-ScoreMode Combinations\n"
        top_results = sorted(
            valid_results,
            key=lambda x: x.correlation,
            reverse=True,
        )[:10]
        markdown += "| Rank | Model | Dataset | Score Mode | Correlation | Success Rate |\n"
        markdown += "|------|-------|---------|------------|-------------|---------------|\n"

        for i, result in enumerate(top_results, 1):
            model_short = (
                result.model_name.split("/")[-1]
                if "/" in result.model_name
                else result.model_name
            )
            success_str = (
                f"{result.success_rate:.3f}"
                if result.success_rate is not None
                else "N/A"
            )
            markdown += f"| {i} | {model_short} | {result.dataset} | {result.score_mode.value} | {result.correlation:.4f} | {success_str} |\n"

    if len(score_modes) >= 2 and valid_results:
        markdown += "\n### Score Mode Analysis\n"
        amean_results = [
            r for r in valid_results if r.score_mode == ScoreMode.AMEAN
        ]
        hmean_results = [
            r for r in valid_results if r.score_mode == ScoreMode.HMEAN
        ]

        if amean_results and hmean_results:
            amean_avg = sum(
                r.correlation
                for r in amean_results
                if r.correlation is not None
            ) / len(
                amean_results,
            )
            hmean_avg = sum(
                r.correlation
                for r in hmean_results
                if r.correlation is not None
            ) / len(
                hmean_results,
            )

            better_mode = "AMEAN" if amean_avg > hmean_avg else "HMEAN"
            diff = abs(amean_avg - hmean_avg)

            markdown += f"- **{better_mode}** performs better on average by {diff:.4f}\n"

            dataset_preferences = {}
            datasets = list(set(r.dataset for r in results))
            for dataset in datasets:
                dataset_amean = [
                    r for r in amean_results if r.dataset == dataset
                ]
                dataset_hmean = [
                    r for r in hmean_results if r.dataset == dataset
                ]

                if dataset_amean and dataset_hmean:
                    amean_best = max(
                        dataset_amean,
                        key=lambda x: x.correlation,
                    )
                    hmean_best = max(
                        dataset_hmean,
                        key=lambda x: x.correlation,
                    )

                    if amean_best.correlation > hmean_best.correlation:
                        dataset_preferences[dataset] = (
                            "AMEAN",
                            amean_best.correlation - hmean_best.correlation,
                        )
                    else:
                        dataset_preferences[dataset] = (
                            "HMEAN",
                            hmean_best.correlation - amean_best.correlation,
                        )

            if dataset_preferences:
                markdown += "- **Dataset-specific preferences**:\n"
                for dataset, (
                    preferred_mode,
                    diff,
                ) in dataset_preferences.items():
                    markdown += (
                        f"  - {dataset}: {preferred_mode} (+{diff:.4f})\n"
                    )

    error_results = [r for r in results if r.error_count > 0]
    if error_results:
        markdown += "\n## Error Analysis\n\n"
        total_errors = sum(r.error_count for r in error_results)
        total_attempts = sum(r.total_count for r in results)
        overall_error_rate = (
            total_errors / total_attempts if total_attempts > 0 else 0
        )

        markdown += f"- **Overall Error Rate**: {overall_error_rate:.3f} ({total_errors}/{total_attempts})\n"

        for result in error_results:
            error_rate = (
                result.error_count / result.total_count
                if result.total_count > 0
                else 0
            )
            markdown += f"- **{result.dataset} ({result.score_mode.value})**: {error_rate:.3f} error rate ({result.error_count}/{result.total_count})\n"

    return markdown


def save_nli_tuning_results(
    results: List[NLIBenchmarkResult],
    output_dir: Optional[Path] = None,
) -> None:
    """Save NLI benchmark results to files."""
    if output_dir is None:
        output_dir = Path("results/nli_benchmark")

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = (
        output_dir / f"nli_benchmark_{timestamp}_{results[0].threshold}.md"
    )
    markdown_content = generate_nli_report(results)

    with open(results_file, "w") as f:
        f.write(markdown_content)

    latest_file = Path("NLI_BENCHMARK_RESULTS.md")
    with open(latest_file, "w") as f:
        f.write(markdown_content)

    summary_file = (
        output_dir / f"summary_{timestamp}_{results[0].threshold}.txt"
    )
    with open(summary_file, "w") as f:
        f.write("NLI Benchmark Summary\n")
        f.write("=" * 30 + "\n\n")

        valid_results = [r for r in results if r.correlation is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x.correlation)
            f.write("Best Configuration:\n")
            f.write(f"  Dataset: {best.dataset}\n")
            f.write(f"  Score Mode: {best.score_mode.value}\n")
            f.write(f"  Metric: {best.metric}\n")
            f.write(f"  Correlation: {best.correlation:.4f}\n")
            f.write(f"  Success Rate: {best.success_rate:.3f}\n")
            f.write(f"  Sample Size: {best.sample_size}\n")
            f.write(f"  Model: {best.model_name}\n")
            f.write(
                f"  Ground Truth Threshold: {best.ground_truth_threshold}\n",
            )
            f.write("\n")

            f.write("All Results (sorted by correlation):\n")
            sorted_results = sorted(
                valid_results,
                key=lambda x: x.correlation,
                reverse=True,
            )
            for result in sorted_results:
                f.write(
                    f"  {result.dataset} | {result.score_mode.value} | {result.metric} | threshold={result.threshold} | gt={result.ground_truth_threshold} | {result.correlation:.4f} | {result.success_rate:.3f}\n",
                )

        # Score mode comparison
        score_modes = list(set(r.score_mode for r in valid_results))
        if len(score_modes) >= 2:
            f.write("\nScore Mode Comparison:\n")
            for mode in score_modes:
                mode_results = [
                    r for r in valid_results if r.score_mode == mode
                ]
                if mode_results:
                    avg_corr = sum(
                        r.correlation
                        for r in mode_results
                        if r.correlation is not None
                    ) / len(
                        mode_results,
                    )
                    f.write(
                        f"  {mode.value}: {avg_corr:.4f} avg ({len(mode_results)} evals)\n",
                    )

    logger.info(f"Saved NLI benchmark results to {results_file}")
    logger.info(f"Saved latest results to {latest_file}")
    logger.info(f"Saved summary to {summary_file}")
