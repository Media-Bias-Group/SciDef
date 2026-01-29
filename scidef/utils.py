"""
Utility functions: text processing, storage, and visualization.
"""

import csv
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from scidef.model.dataclass import (
    EvaluationResult,
    ExtractionResult,
    SimilarityResult,
)


class ColorFormatter(logging.Formatter):
    # ANSI escape codes
    RESET = "\x1b[0m"
    COLORS = {
        logging.DEBUG: "\x1b[36m",  # cyan
        logging.INFO: "\x1b[32m",  # green
        logging.WARNING: "\x1b[33m",  # yellow
        logging.ERROR: "\x1b[31m",  # red
        logging.CRITICAL: "\x1b[35m",  # magenta
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{msg}{self.RESET}"


def get_custom_colored_logger(name: str) -> logging.Logger:
    """Get a logger with colored output."""
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if this is called repeatedly
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(
            ColorFormatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            ),
        )
        logger.addHandler(console_handler)

    # IMPORTANT: prevent records from bubbling up to root
    logger.propagate = False

    return logger


logger = get_custom_colored_logger(__name__)


# Storage Functions
def save_extraction_results_to_csv(
    results: List[ExtractionResult],
    output_path: Path,
    delimiter: str = "|",
    quotechar: str = '"',
) -> None:
    """Save extraction results to CSV file."""
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for result in results:
        base_record = {
            "paper_id": result.paper_id,
            "mode": result.mode.value,
            "status": result.status.value,
            "error_message": result.error_message or "",
            "thought_process": result.thought_process or "",
            "total_duration": result.metrics.total_duration,
            "definition_count": len(result.definitions),
        }

        if result.definitions:
            for i, definition in enumerate(result.definitions):
                record = base_record.copy()
                record.update(
                    {
                        "definition_index": i,
                        "concept": definition.concept,
                        "definition_text": definition.definition_text,
                    },
                )
                records.append(record)
        else:
            base_record["definition_index"] = -1
            base_record["concept"] = ""
            base_record["definition_text"] = ""
            records.append(base_record)

    df = pd.DataFrame(records)
    df.to_csv(
        output_path,
        index=False,
        sep=delimiter,
        quotechar=quotechar,
        quoting=csv.QUOTE_MINIMAL,
    )
    logger.info(f"Saved {len(records)} extraction records to {output_path}")


def save_evaluation_results_to_csv(
    results: List[EvaluationResult],
    output_path: Path,
    delimiter: str = "|",
    quotechar: str = '"',
) -> None:
    """Save evaluation results to CSV file."""
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for eval_result in results:
        paper_id = eval_result.paper_id

        # Similarity results
        for sim_result in eval_result.similarity_results:
            record = {
                "paper_id": paper_id,
                "evaluation_type": "similarity",
                "human_concept": sim_result.human_concept,
                "model_mode": sim_result.model_mode.value,
                "similarity_score": sim_result.similarity_score,
                "error_message": sim_result.embedding_error,
            }
            all_records.append(record)

        # Judge results
        for judge_result in eval_result.judge_results:
            record = {
                "paper_id": paper_id,
                "evaluation_type": "judge",
                "human_concept": judge_result.human_concept,
                "model_mode": judge_result.model_mode.value,
                "judgment_category": judge_result.judgment_category
                if judge_result.judgment_category
                else None,
                "error_message": judge_result.judgment_error,
            }
            all_records.append(record)

        # NLI results
        for nli_result in eval_result.nli_results:
            record = {
                "paper_id": paper_id,
                "evaluation_type": "nli",
                "human_concept": nli_result.human_concept,
                "model_mode": nli_result.model_mode.value,
                "entailment_score": nli_result.entailment_score,
                "predicted_label": nli_result.predicted_label,
                "error_message": nli_result.nli_error,
            }
            all_records.append(record)

    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(
            output_path,
            index=False,
            sep=delimiter,
            quotechar=quotechar,
            quoting=csv.QUOTE_MINIMAL,
        )
        logger.info(
            f"Saved {len(all_records)} evaluation records to {output_path}",
        )


# Visualization Functions
def generate_similarity_histogram(
    similarity_results: List[SimilarityResult],
    output_path: Path,
) -> None:
    """Generate histogram of similarity scores."""
    if not PLOTTING_AVAILABLE:
        logger.warning(
            "Plotting not available - skipping similarity histogram",
        )
        return

    scores = [
        r.similarity_score
        for r in similarity_results
        if r.similarity_score is not None
    ]

    if not scores:
        logger.warning("No valid similarity scores to plot")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor="black", alpha=0.7)
    plt.axvline(
        np.mean(scores),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(scores):.3f}",
    )
    plt.axvline(
        np.median(scores),
        color="orange",
        linestyle="--",
        label=f"Median: {np.median(scores):.3f}",
    )
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Similarity Scores (n={len(scores)})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved similarity histogram to {output_path}")


def generate_performance_summary(
    evaluation_results: List[EvaluationResult],
    output_path: Path,
) -> None:
    """Generate a comprehensive performance summary plot."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Plotting not available - skipping performance summary")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Collect all data
    similarity_scores = []
    judge_categories = []
    nli_scores = []

    for eval_result in evaluation_results:
        similarity_scores.extend(
            [
                r.similarity_score
                for r in eval_result.similarity_results
                if r.similarity_score is not None
            ],
        )
        judge_categories.extend(
            [
                r.judgment_category
                for r in eval_result.judge_results
                if r.judgment_category
            ],
        )
        nli_scores.extend(
            [
                r.entailment_score
                for r in eval_result.nli_results
                if r.entailment_score is not None
            ],
        )

    # Similarity histogram
    if similarity_scores:
        axes[0, 0].hist(
            similarity_scores,
            bins=20,
            edgecolor="black",
            alpha=0.7,
        )
        axes[0, 0].axvline(
            np.mean(similarity_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(similarity_scores):.3f}",
        )
        axes[0, 0].set_xlabel("Similarity Score")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].set_title(f"Similarity Scores (n={len(similarity_scores)})")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

    # Judge categories
    if judge_categories:
        category_counts = pd.Series(judge_categories).value_counts()
        axes[0, 1].bar(category_counts.index, category_counts.values)
        axes[0, 1].set_xlabel("Judgment Category")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title(f"Judge Categories (n={len(judge_categories)})")
        axes[0, 1].tick_params(axis="x", rotation=45)

    # NLI scores
    if nli_scores:
        axes[1, 0].hist(nli_scores, bins=20, edgecolor="black", alpha=0.7)
        axes[1, 0].axvline(
            np.mean(nli_scores),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(nli_scores):.3f}",
        )
        axes[1, 0].set_xlabel("Entailment Score")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title(f"NLI Entailment Scores (n={len(nli_scores)})")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # Summary statistics
    stats_text = f"""Summary Statistics:

Similarity (n={len(similarity_scores)}):
  Mean: {np.mean(similarity_scores):.3f if similarity_scores else 'N/A'}
  Std:  {np.std(similarity_scores):.3f if similarity_scores else 'N/A'}

NLI (n={len(nli_scores)}):
  Mean: {np.mean(nli_scores):.3f if nli_scores else 'N/A'}
  Std:  {np.std(nli_scores):.3f if nli_scores else 'N/A'}

Judge (n={len(judge_categories)}):
  Most Common: {pd.Series(judge_categories).mode()[0] if judge_categories else "N/A"}
"""

    axes[1, 1].text(
        0.1,
        0.9,
        stats_text,
        transform=axes[1, 1].transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=10,
    )
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis("off")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved performance summary to {output_path}")
