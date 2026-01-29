from datetime import datetime
from pathlib import Path
from typing import List

from scidef.benchmark import BenchmarkResult
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


def generate_markdown_results(results: List[BenchmarkResult]) -> str:
    """Generate markdown formatted benchmark results."""
    if not results:
        return "# Benchmark Results\n\nNo results available.\n"

    markdown = "# Benchmark Results\n\n"
    markdown += (
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
    )

    # Summary table
    markdown += "## Results Summary\n\n"
    markdown += "| Metric | Dataset | Model | Prompt | System Prompt | Correlation | Success Rate | Errors | Total |\n"
    markdown += "|--------|---------|-------|--------|---------------|-------------|--------------|--------|-------|\n"

    for result in results:
        correlation_str = (
            f"{result.correlation:.3f}"
            if result.correlation is not None
            else "N/A"
        )
        success_rate_str = f"{result.success_rate:.1%}"
        model_name = result.model_name or "Unknown"
        prompt_type = result.prompt_type or "-"
        system_prompt = result.system_prompt or "-"

        markdown += f"| {result.metric} | {result.dataset} | {model_name} | {prompt_type} | {system_prompt} | {correlation_str} | {success_rate_str} | {result.error_count} | {result.total_count} |\n"

    # Detailed breakdown by dataset
    datasets = list(set(r.dataset for r in results))
    for dataset in sorted(datasets):
        dataset_results = [r for r in results if r.dataset == dataset]
        if not dataset_results:
            continue

        markdown += f"\n## {dataset.upper()} Dataset\n\n"

        for result in dataset_results:
            model_name = result.model_name or "Unknown"
            prompt_info = (
                f" - {result.prompt_type}" if result.prompt_type else ""
            )
            markdown += f"### {result.metric} ({model_name}{prompt_info})\n"
            if result.correlation is not None:
                markdown += f"- **Correlation**: {result.correlation:.3f}\n"
            markdown += f"- **Success Rate**: {result.success_rate:.1%}\n"
            markdown += f"- **Successful Evaluations**: {result.total_count - result.error_count}/{result.total_count}\n"
            if result.error_count > 0:
                markdown += f"- **Error Rate**: {result.error_count / result.total_count:.1%}\n"
            if result.system_prompt and result.system_prompt != "-":
                markdown += (
                    f"- **System Prompt Type**: {result.system_prompt}\n"
                )
            markdown += "\n"

    # Performance overview
    markdown += "## Performance Overview\n\n"

    # Best correlations
    corr_results = [r for r in results if r.correlation is not None]
    if corr_results:
        best_corr = max(corr_results, key=lambda x: x.correlation or 0.0)
        model_name = best_corr.model_name or "Unknown"
        prompt_info = (
            f", {best_corr.prompt_type}" if best_corr.prompt_type else ""
        )
        markdown += f"**Best Correlation**: {best_corr.correlation:.3f} ({best_corr.metric} on {best_corr.dataset}, {model_name}{prompt_info})\n\n"

    # Success rates
    best_success = max(results, key=lambda x: x.success_rate)
    model_name = best_success.model_name or "Unknown"
    prompt_info = (
        f", {best_success.prompt_type}" if best_success.prompt_type else ""
    )
    markdown += f"**Highest Success Rate**: {best_success.success_rate:.1%} ({best_success.metric} on {best_success.dataset}, {model_name}{prompt_info})\n\n"

    # Error summary
    total_evals = sum(r.total_count for r in results)
    total_errors = sum(r.error_count for r in results)
    overall_success = (
        (total_evals - total_errors) / total_evals if total_evals > 0 else 0
    )
    markdown += f"**Overall Success Rate**: {overall_success:.1%} ({total_evals - total_errors}/{total_evals} successful evaluations)\n\n"

    return markdown


def parse_existing_results(markdown_file: Path) -> List[BenchmarkResult]:
    """Parse existing benchmark results from markdown file."""
    if not markdown_file.exists():
        return []

    try:
        content = markdown_file.read_text()
        header = "| Metric | Dataset | Model | Prompt | System Prompt | Correlation | Success Rate | Errors | Total |\n"
        header_pos = content.find(header)

        if header_pos == -1:
            return []

        lines = content[header_pos:].split("\n")
        if len(lines) < 3:  # header + separator + at least one data row
            return []

        results = []
        for line in lines[2:]:
            if not line.startswith("|"):
                continue
            cols = [col.strip() for col in line.split("|")[1:-1]]
            if len(cols) not in (8, 9):
                continue
            system_prompt = None
            if len(cols) == 8:
                (
                    metric,
                    dataset,
                    model,
                    prompt,
                    corr_str,
                    success_str,
                    errors_str,
                    total_str,
                ) = cols
            elif len(cols) == 9:
                (
                    metric,
                    dataset,
                    model,
                    prompt,
                    system_prompt_str,
                    corr_str,
                    success_str,
                    errors_str,
                    total_str,
                ) = cols
                system_prompt = (
                    system_prompt_str if system_prompt_str != "-" else None
                )
            correlation = float(corr_str) if corr_str != "N/A" else None
            success_rate = float(success_str.rstrip("%")) / 100.0
            error_count = int(errors_str)
            total_count = int(total_str)
            model_name = model if model != "Unknown" else None
            prompt_type = prompt if prompt != "-" else None
            results.append(
                BenchmarkResult(
                    metric=metric,
                    dataset=dataset,
                    correlation=correlation,
                    success_rate=success_rate,
                    error_count=error_count,
                    total_count=total_count,
                    model_name=model_name,
                    prompt_type=prompt_type,
                    system_prompt=system_prompt,
                ),
            )

        return results

    except Exception as e:
        logger.warning(f"Failed to parse existing results: {e}")
        return []


def merge_results(
    existing_results: List[BenchmarkResult],
    new_results: List[BenchmarkResult],
) -> List[BenchmarkResult]:
    """Merge existing results with new results, keeping old ones for metrics not being run.
    Merge criteria: metric, dataset, model_name, prompt_type, system_prompt
    """
    new_keys = {
        (
            r.metric,
            r.dataset,
            r.model_name,
            r.prompt_type or "",
            r.system_prompt or "",
        )
        for r in new_results
    }
    merged = [
        r
        for r in existing_results
        if (
            r.metric,
            r.dataset,
            r.model_name,
            r.prompt_type or "",
            r.system_prompt or "",
        )
        not in new_keys
    ] + new_results
    return sorted(
        merged,
        key=lambda r: (
            r.metric,
            r.dataset,
            r.model_name,
            r.prompt_type or "",
            r.system_prompt or "",
        ),
    )


def save_benchmark_results(
    results: List[BenchmarkResult],
    output_dir: Path,
) -> None:
    """Save benchmark results, preserving existing results for metrics not being run."""
    output_dir.mkdir(parents=True, exist_ok=True)

    root_markdown_file = Path("BENCHMARK_RESULTS.md")
    existing_results = parse_existing_results(root_markdown_file)
    merged_results = merge_results(existing_results, results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"benchmark_summary_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write("Benchmark Results\n")
        f.write("=" * 30 + "\n\n")

        for result in results:
            f.write(f"{result.metric} on {result.dataset}:\n")
            if result.correlation is not None:
                f.write(f"  Correlation: {result.correlation:.3f}\n")
            f.write(f"  Success rate: {result.success_rate:.1%}\n")
            f.write(f"  Errors: {result.error_count}/{result.total_count}\n\n")

    markdown_content = generate_markdown_results(merged_results)

    with open(root_markdown_file, "w") as f:
        f.write(markdown_content)

    logger.info(f"Saved benchmark summary to {summary_file}")
    logger.info(
        f"Saved merged benchmark markdown to root: {root_markdown_file}",
    )
    logger.info(
        f"Merged {len(existing_results)} existing results with {len(results)} new results",
    )
