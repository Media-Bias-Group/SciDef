from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


@dataclass
class EmbeddingBenchmarkResult:
    """Result from embedding benchmark evaluation."""

    dataset: str
    threshold: Union[float, List[float]]
    metric: str = field(default="")
    ground_truth_threshold: Optional[Union[float, List[float]]] = None
    correlation: Optional[float] = None
    sample_size: int | str = 0
    split: str = "train"
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not self.metric:
            raise ValueError(
                "metric must be provided for EmbeddingBenchmarkResult",
            )
        if self.timestamp is None:
            self.timestamp = datetime.now()


def generate_embedding_report(results: List[EmbeddingBenchmarkResult]) -> str:
    """Generate a simple markdown report for embedding tuning results."""
    if not results:
        return "# Embedding BenchmarkResults\n\nNo results available.\n"

    markdown = "# Embedding Benchmark Results\n\n"
    markdown += (
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    )

    # Summary table
    markdown += "## Results Summary\n\n"
    markdown += "| Dataset | Metric | Threshold | GT Threshold | Result | Sample Size | Split |\n"
    markdown += "|---------|--------|-----------|--------------|-------------|-------------|-------|\n"

    # Sort results by dataset then metric for organized viewing
    sorted_results = sorted(results, key=lambda x: (x.dataset, x.metric))

    for result in sorted_results:
        correlation_str = (
            f"{result.correlation:.4f}"
            if result.correlation is not None
            else "N/A"
        )
        threshold_str = (
            str(result.threshold)
            if isinstance(result.threshold, float)
            else f"[{', '.join(map(str, result.threshold))}]"
        )
        gt_threshold_str = (
            str(result.ground_truth_threshold)
            if isinstance(result.ground_truth_threshold, float)
            else (
                f"[{', '.join(map(str, result.ground_truth_threshold))}]"
                if isinstance(result.ground_truth_threshold, list)
                else "None"
            )
        )
        markdown += (
            f"| {result.dataset} | {result.metric} | {threshold_str} | {gt_threshold_str} | "
            f"{correlation_str} | {result.sample_size} | {result.split} |\n"
        )

    markdown += "\n## Best Configurations\n\n"

    datasets = list(set(r.dataset for r in results))
    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r.dataset == dataset]
        valid_results = [
            r for r in dataset_results if r.correlation is not None
        ]

        if not valid_results:
            continue

        best_result = max(valid_results, key=lambda x: x.correlation)
        threshold_str = (
            str(best_result.threshold)
            if isinstance(best_result.threshold, float)
            else f"[{', '.join(map(str, best_result.threshold))}] (bucketing)"
        )
        markdown += f"### {dataset.upper()} Dataset\n"
        markdown += f"- **Best Threshold**: {threshold_str}\n"
        if best_result.ground_truth_threshold is not None:
            gt_threshold_str = (
                str(best_result.ground_truth_threshold)
                if isinstance(best_result.ground_truth_threshold, float)
                else f"[{', '.join(map(str, best_result.ground_truth_threshold))}]"
            )
            markdown += f"- **Ground Truth Threshold**: {gt_threshold_str}\n"
        markdown += f"- **Best Metric**: {best_result.metric}\n"
        markdown += f"- **Result**: {best_result.correlation:.4f}\n"
        markdown += f"- **Sample Size**: {best_result.sample_size}\n\n"

    return markdown


def save_embedding_tuning_results(
    results: List[EmbeddingBenchmarkResult],
    output_dir: Optional[Path] = None,
) -> None:
    """Save embedding tuning results to files."""
    if output_dir is None:
        output_dir = Path("results/embedding_benchmark")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"embedding_benchmark_{timestamp}.md"
    markdown_content = generate_embedding_report(results)

    with open(results_file, "w") as f:
        f.write(markdown_content)

    latest_file = Path("EMBEDDING_BENCHMARK_RESULTS.md")
    with open(latest_file, "w") as f:
        f.write(markdown_content)

    summary_file = output_dir / f"summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("Embedding Tuning Summary\n")
        f.write("=" * 30 + "\n\n")

        valid_results = [r for r in results if r.correlation is not None]
        if valid_results:
            best = max(valid_results, key=lambda x: x.correlation)
            threshold_str = (
                str(best.threshold)
                if isinstance(best.threshold, float)
                else f"[{', '.join(map(str, best.threshold))}] (bucketing)"
            )
            f.write("Best Configuration:\n")
            f.write(f"  Dataset: {best.dataset}\n")
            f.write(f"  Threshold: {threshold_str}\n")
            if best.ground_truth_threshold is not None:
                gt_threshold_str = (
                    str(best.ground_truth_threshold)
                    if isinstance(best.ground_truth_threshold, float)
                    else f"[{', '.join(map(str, best.ground_truth_threshold))}]"
                )
                f.write(f"  Ground Truth Threshold: {gt_threshold_str}\n")
            f.write(f"  Metric: {best.metric}\n")
            f.write(f"  Correlation: {best.correlation:.4f}\n")
            f.write(f"  Sample Size: {best.sample_size}\n\n")

            f.write("All Results (sorted by correlation):\n")
            sorted_results = sorted(
                valid_results,
                key=lambda x: x.correlation,
                reverse=True,
            )
            for result in sorted_results:
                threshold_str = (
                    str(result.threshold)
                    if isinstance(result.threshold, float)
                    else f"[{', '.join(map(str, result.threshold))}]"
                )
                gt_threshold_str = (
                    str(result.ground_truth_threshold)
                    if isinstance(result.ground_truth_threshold, float)
                    else (
                        f"[{', '.join(map(str, result.ground_truth_threshold))}]"
                        if isinstance(result.ground_truth_threshold, list)
                        else "None"
                    )
                )
                f.write(
                    f"  {result.dataset} | {result.metric} | {threshold_str} | gt={gt_threshold_str} | {result.correlation:.4f}\n",
                )

    logger.info(f"Saved embedding benchmark results to {results_file}")
    logger.info(f"Saved latest results to {latest_file}")
    logger.info(f"Saved summary to {summary_file}")
