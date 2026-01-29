from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from scidef.model.dataclass import JudgeSystemPrompt
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


@dataclass
class JudgeBenchmarkResult:
    """Result from LLM judge benchmark evaluation."""

    dataset: str
    system_prompt: JudgeSystemPrompt
    temperature: float
    top_p: float
    threshold: Optional[Union[float, List[float]]]
    metric: str = field(default="")
    ground_truth_threshold: Optional[Union[float, List[float]]] = None
    correlation: Optional[float] = None
    sample_size: int = 0
    split: str = "train"
    success_rate: Optional[float] = None
    error_count: int = 0
    total_count: int = 0
    model_name: str = "Unknown"
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not self.metric:
            raise ValueError("metric must be provided for JudgeBenchmarkResult")
        if self.timestamp is None:
            self.timestamp = datetime.now()


def generate_judge_report(results: List[JudgeBenchmarkResult]) -> str:
    """Generate a comprehensive markdown report for LLM judge tuning results."""
    if not results:
        return "# LLM Judge Benchmark Results\n\nNo results available.\n"

    markdown = "# LLM Judge Benchmark Results\n\n"
    markdown += (
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    )

    models = list(
        set(r.model_name for r in results if r.model_name != "Unknown"),
    )
    if models:
        markdown += f"**Models Evaluated**: {', '.join(models)}\n\n"

    valid_results = [r for r in results if r.correlation is not None]
    if valid_results:
        correlations = [
            r.correlation for r in valid_results if r.correlation is not None
        ]
        avg_correlation = sum(correlations) / len(correlations)
        best_correlation = max(correlations)
        markdown += "## Summary Statistics\n\n"
        markdown += f"- **Total Evaluations**: {len(results)}\n"
        markdown += f"- **Successful Evaluations**: {len(valid_results)}\n"
        markdown += f"- **Average Correlation**: {avg_correlation:.4f}\n"
        markdown += f"- **Best Correlation**: {best_correlation:.4f}\n\n"

    markdown += "## Detailed Results\n\n"
    markdown += "| Dataset | Judge Prompt  | Temp | Top-p | threshold | GT Threshold | Correlation | Success Rate | Sample Size | Split | Model | Metric |\n"
    markdown += "|---------|--------------|------|-------|---------|---------------|-------------|--------------|-------------|-------|-------|--------|\n"

    # Sort results by correlation (descending) then by dataset
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
            f"| {result.dataset} | {result.system_prompt.value} | "
            f"{result.temperature} | {result.top_p} | {result.threshold} | "
            f"{result.ground_truth_threshold} | {correlation_str} | {success_rate_str} | "
            f"{result.sample_size} | {result.split} | {result.model_name} | {result.metric} |\n"
        )

    markdown += "\n## Best Configurations by Dataset\n\n"
    datasets = list(set(r.dataset for r in results))

    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r.dataset == dataset]
        valid_dataset_results = [
            r for r in dataset_results if r.correlation is not None
        ]

        if not valid_dataset_results:
            continue

        best_result = max(valid_dataset_results, key=lambda x: x.correlation)
        markdown += f"### {dataset.upper()} Dataset\n"
        markdown += f"- **System Prompt**: {best_result.system_prompt.value}\n"
        markdown += f"- **Temperature**: {best_result.temperature}\n"
        markdown += f"- **Top-p**: {best_result.top_p}\n"
        markdown += f"- **Thresholds**: {best_result.threshold}\n"
        if best_result.ground_truth_threshold is not None:
            markdown += f"- **Ground Truth Thresholds**: {best_result.ground_truth_threshold}\n"
        markdown += f"- **Correlation**: {best_result.correlation:.4f}\n"
        markdown += (
            f"- **Success Rate**: {best_result.success_rate:.3f}\n"
            if best_result.success_rate
            else ""
        )
        markdown += f"- **Sample Size**: {best_result.sample_size}\n"
        markdown += f"- **Model**: {best_result.model_name}\n\n"

    markdown += "## System Prompt Performance Analysis\n\n"
    system_prompts = list(set(r.system_prompt for r in results))

    for prompt in sorted(system_prompts, key=lambda x: x.value):
        prompt_results = [
            r
            for r in results
            if r.system_prompt == prompt and r.correlation is not None
        ]

        if prompt_results:
            correlations = [
                r.correlation
                for r in prompt_results
                if r.correlation is not None
            ]
            avg_corr = sum(correlations) / len(correlations)
            max_corr = max(correlations)
            min_corr = min(correlations)

            markdown += f"### {prompt.value}\n"
            markdown += f"- **Average Correlation**: {avg_corr:.4f}\n"
            markdown += f"- **Best Correlation**: {max_corr:.4f}\n"
            markdown += f"- **Worst Correlation**: {min_corr:.4f}\n"
            markdown += f"- **Evaluations**: {len(prompt_results)}\n\n"

    markdown += "## Temperature Impact Analysis\n\n"
    temperatures = sorted(list(set(r.temperature for r in results)))

    for temp in temperatures:
        temp_results = [
            r
            for r in results
            if r.temperature == temp and r.correlation is not None
        ]

        if temp_results:
            correlations = [
                r.correlation
                for r in temp_results
                if r.correlation is not None
            ]
            avg_corr = sum(correlations) / len(correlations)
            max_corr = max(correlations)
            min_corr = min(correlations)

            markdown += f"### Temperature = {temp}\n"
            markdown += f"- **Average Correlation**: {avg_corr:.4f}\n"
            markdown += f"- **Best Correlation**: {max_corr:.4f}\n"
            markdown += f"- **Worst Correlation**: {min_corr:.4f}\n"
            markdown += f"- **Evaluations**: {len(temp_results)}\n\n"

    markdown += "## Top-p Impact Analysis\n\n"
    top_p_values = sorted(list(set(r.top_p for r in results)))

    for top_p in top_p_values:
        top_p_results = [
            r
            for r in results
            if r.top_p == top_p and r.correlation is not None
        ]

        if top_p_results:
            correlations = [
                r.correlation
                for r in top_p_results
                if r.correlation is not None
            ]
            avg_corr = sum(correlations) / len(correlations)
            max_corr = max(correlations)
            min_corr = min(correlations)

            markdown += f"### Top-p = {top_p}\n"
            markdown += f"- **Average Correlation**: {avg_corr:.4f}\n"
            markdown += f"- **Best Correlation**: {max_corr:.4f}\n"
            markdown += f"- **Worst Correlation**: {min_corr:.4f}\n"
            markdown += f"- **Evaluations**: {len(top_p_results)}\n\n"

    markdown += "## Performance Insights\n\n"

    if valid_results:
        markdown += "### Top Configuration Combinations\n"
        top_results = sorted(
            valid_results,
            key=lambda x: x.correlation,
            reverse=True,
        )[:10]
        markdown += "| Rank | Dataset | Judge Prompt | System Prompt | Temp | Top-p | Correlation | Success Rate |\n"
        markdown += "|------|---------|--------------|---------------|------|-------|-------------|---------------|\n"

        for i, result in enumerate(top_results, 1):
            success_str = (
                f"{result.success_rate:.3f}"
                if result.success_rate is not None
                else "N/A"
            )
            markdown += f"| {i} | {result.dataset} | {result.system_prompt.value} | {result.temperature} | {result.top_p} | {result.correlation:.4f} | {success_str} |\n"

        markdown += "\n### Parameter Recommendations\n"

        temp_avg_scores = {}
        for temp in temperatures:
            temp_results = [r for r in valid_results if r.temperature == temp]
            if temp_results:
                correlations = [
                    r.correlation
                    for r in temp_results
                    if r.correlation is not None
                ]
                temp_avg_scores[temp] = sum(correlations) / len(correlations)

        if temp_avg_scores:
            best_temp = max(temp_avg_scores, key=lambda x: temp_avg_scores[x])
            markdown += f"- **Recommended Temperature**: {best_temp} (avg correlation: {temp_avg_scores[best_temp]:.4f})\n"

        top_p_avg_scores = {}
        for top_p in top_p_values:
            top_p_results = [r for r in valid_results if r.top_p == top_p]
            if top_p_results:
                correlations = [
                    r.correlation
                    for r in top_p_results
                    if r.correlation is not None
                ]
                top_p_avg_scores[top_p] = sum(correlations) / len(correlations)

        if top_p_avg_scores:
            best_top_p = max(
                top_p_avg_scores,
                key=lambda x: top_p_avg_scores[x],
            )
            markdown += f"- **Recommended Top-p**: {best_top_p} (avg correlation: {top_p_avg_scores[best_top_p]:.4f})\n"

    error_results = [r for r in results if r.error_count > 0]
    if error_results:
        markdown += "\n## Error Analysis\n\n"
        total_errors = sum(r.error_count for r in error_results)
        total_attempts = sum(r.total_count for r in results)
        overall_error_rate = (
            total_errors / total_attempts if total_attempts > 0 else 0
        )

        markdown += f"- **Overall Error Rate**: {overall_error_rate:.3f} ({total_errors}/{total_attempts})\n"

        config_errors = {}
        for result in error_results:
            config = f"{result.system_prompt.value} (T={result.temperature}, p={result.top_p})"
            error_rate = (
                result.error_count / result.total_count
                if result.total_count > 0
                else 0
            )
            config_errors[config] = error_rate

        worst_configs = sorted(
            config_errors.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        markdown += "\n**Worst Error Rates by Configuration:**\n"
        for config, error_rate in worst_configs:
            markdown += f"- **{config}**: {error_rate:.3f} error rate\n"

    return markdown


def save_judge_tuning_results(
    results: List[JudgeBenchmarkResult],
    output_dir: Optional[Path] = None,
) -> None:
    """Save LLM judge tuning results to files."""
    if output_dir is None:
        output_dir = Path("results/judge_benchmark")

    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"judge_benchmark_{timestamp}.md"
    markdown_content = generate_judge_report(results)

    with open(results_file, "w") as f:
        f.write(markdown_content)

    latest_file = Path("JUDGE_BENCHMARK_RESULTS.md")
    with open(latest_file, "w") as f:
        f.write(markdown_content)

    summary_file = output_dir / f"summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("LLM Judge Benchmark Summary\n")
        f.write("=" * 30 + "\n\n")

        valid_results = [r for r in results if r.correlation is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x.correlation)
            f.write("Best Configuration:\n")
            f.write(f"  Dataset: {best.dataset}\n")
            f.write(f"  System Prompt: {best.system_prompt.value}\n")
            f.write(f"  Temperature: {best.temperature}\n")
            f.write(f"  Top-p: {best.top_p}\n")
            f.write(f"  threshold: {best.threshold}\n")
            if best.ground_truth_threshold is not None:
                f.write(
                    f"  Ground Truth Threshold: {best.ground_truth_threshold}\n",
                )
            f.write(f"  Correlation: {best.correlation:.4f}\n")
            f.write(f"  Success Rate: {best.success_rate:.3f}\n")
            f.write(f"  Sample Size: {best.sample_size}\n")
            f.write(f"  Model: {best.model_name}\n\n")

            f.write("All Results (sorted by correlation):\n")
            sorted_results = sorted(
                valid_results,
                key=lambda x: x.correlation,
                reverse=True,
            )
            for result in sorted_results:
                f.write(
                    f"  {result.dataset} | {result.system_prompt.value} | "
                    f"T={result.temperature} | p={result.top_p} | threshold={result.threshold} | "
                    f"gt={result.ground_truth_threshold} | {result.correlation:.4f} | {result.success_rate:.3f}\n",
                )

        f.write("\nParameter Analysis:\n")
        temperatures = sorted(list(set(r.temperature for r in valid_results)))
        if temperatures:
            f.write("Temperature Performance:\n")
            for temp in temperatures:
                temp_results = [
                    r for r in valid_results if r.temperature == temp
                ]
                if temp_results:
                    correlations = [
                        r.correlation
                        for r in temp_results
                        if r.correlation is not None
                    ]
                    avg_corr = sum(correlations) / len(correlations)
                    f.write(
                        f"  {temp}: {avg_corr:.4f} avg ({len(temp_results)} evals)\n",
                    )

        top_p_values = sorted(list(set(r.top_p for r in valid_results)))
        if top_p_values:
            f.write("Top-p Performance:\n")
            for top_p in top_p_values:
                top_p_results = [r for r in valid_results if r.top_p == top_p]
                if top_p_results:
                    correlations = [
                        r.correlation
                        for r in top_p_results
                        if r.correlation is not None
                    ]
                    avg_corr = sum(correlations) / len(correlations)
                    f.write(
                        f"  {top_p}: {avg_corr:.4f} avg ({len(top_p_results)} evals)\n",
                    )

    logger.info(f"Saved LLM benchmark benchmark results to {results_file}")
    logger.info(f"Saved latest results to {latest_file}")
    logger.info(f"Saved summary to {summary_file}")
