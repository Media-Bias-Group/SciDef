import argparse
import math
import os
import statistics
from pathlib import Path

import dspy

import wandb
from config import Config, setup_logging
from scidef.evaluation.utils import evaluate_and_log
from scidef.extraction.dataclass import ChunkMode
from scidef.extraction.extractor import (
    MultiStepExtractor,
    OnesStepExtractor,
)
from scidef.extraction.extractor.dspy_extraction import DSPyPaperExtractor
from scidef.extraction.service import ExtractionService
from scidef.extraction.utils import load_ground_truth, make_splits
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


def get_detailed_stats(rows_list: list, metric_keys: list) -> str:
    """Generates a formatted string of statistics for specific keys in the rows."""
    output = []
    output.append("-" * 30)

    if not rows_list:
        return "No data available."

    for key in metric_keys:
        values = [r[key] for r in rows_list if key in r and r[key] is not None]
        if not values:
            continue

        values.sort()
        n = len(values)
        _min = values[0]
        _max = values[-1]
        _mean = sum(values) / n
        _median = statistics.median(values)

        # Linear interpolation for percentiles
        def get_percentile(p):
            k = (n - 1) * p
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return values[int(k)]
            return values[int(f)] * (c - k) + values[int(c)] * (k - f)

        p10 = get_percentile(0.10)
        p25 = get_percentile(0.25)

        output.append(f"Metric: {key}")
        output.append(f"  Mean:   {_mean:.4f}")
        output.append(f"  Median: {_median:.4f}")
        output.append(f"  Min:    {_min:.4f}")
        output.append(f"  Max:    {_max:.4f}")
        output.append(f"  10%:    {p10:.4f}")
        output.append(f"  25%:    {p25:.4f}")
        output.append("")  # Empty line for spacing

    output.append("-" * 30)
    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Extract definitions from academic papers",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        help="path containing ground truths",
        default="data/definitions/all_concepts_combined.json",
    )
    parser.add_argument(
        "--extractions-dir",
        type=Path,
        nargs="+",
        default=[
            Path("ManualPDFsGROBID/manual_pdfs_grobid"),
            Path("ManualPDFsGROBID/new_grobid"),
        ],
        help="Directory containing XML files",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--llm-model-name",
        type=str,
        default="openai/gpt-oss-20b",
        help="LLM model name for DSPy",
    )

    parser.add_argument(
        "--nli-model-name",
        type=str,
        default="tasksource/ModernBERT-large-nli",
        help="NLI model name",
    )

    parser.add_argument(
        "--nli-threshold",
        type=float,
        default=0.60,
        help="NLI score threshold for correctness",
    )

    parser.add_argument(
        "--allow-out-of-vocab",
        action="store_true",
        help="Allow out-of-vocabulary terms during evaluation",
    )

    parser.add_argument(
        "--load-compiled-path",
        type=str,
        default="",
        help="Path to load compiled DSPy program from (skip optimization)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for LLM responses",
    )

    parser.add_argument(
        "--two-step-extraction",
        action="store_true",
        help="Use two-step extraction process (i.e., first determine if section contains definitions, only then extract).",
    )

    parser.add_argument(
        "--chunk-mode",
        choices=[m.value for m in ChunkMode],
        default=ChunkMode.SECTION.value,
        help="Processing modes",
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip evaluation on training set",
    )
    parser.add_argument(
        "--skip-dev",
        action="store_true",
        help="Skip evaluation on dev set",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip evaluation on test set",
    )

    parser.add_argument(
        "--extractor-type",
        choices=["PaperExtractor", "MultiStepExtractor", "OneStepExtractor"],
        default="PaperExtractor",
        help="Type of extractor to use (DSPy or custom extractors)",
    )

    parser.add_argument(
        "--extracted-definitions-path",
        type=Path,
        help="Path to extracted definitions JSON file (this will be used for evaluation instead of running extraction anew).",
    )

    parser.add_argument(
        "--nli-compile",
        action="store_true",
        help="Compile NLI model for faster inference",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger.info("Starting evaluation...")
    logger.info(f"Ground truth path: {args.ground_truth_path}")
    logger.info(f"Extractions dir: {args.extractions_dir}")
    logger.info(f"LLM model name: {args.llm_model_name}")
    logger.info(f"NLI model name: {args.nli_model_name}")
    logger.info(f"NLI threshold: {args.nli_threshold}")
    logger.info(
        f"Allow out-of-vocab terms: {args.allow_out_of_vocab}",
    )
    logger.info(
        f"Load compiled DSPy program path: {args.load_compiled_path or 'N/A'}",
    )
    logger.info(
        f"Use two-step extraction process: {args.two_step_extraction}",
    )
    logger.info(
        f"Extracted definitions path: {args.extracted_definitions_path or 'N/A'}",
    )
    logger.info(f"Chunk mode: {args.chunk_mode}")
    logger.info(f"NLI Compilation: {args.nli_compile}")

    # Setup NLI model
    config = Config()
    nli_client = config.create_nli_client(
        model_name=args.nli_model_name,
        compile=args.nli_compile,
    )
    if not nli_client:
        logger.error(
            f"Failed to create NLI client for model: {args.nli_model_name}. Skipping...",
        )
        return

    # ensure name is right in env for cache
    cleaned_model_name = "/".join(args.llm_model_name.split("/")[1:])
    os.environ["LLM_MODEL_NAME"] = cleaned_model_name
    os.environ["VLLM_MODEL_NAME"] = cleaned_model_name
    os.environ["OPENROUTER_MODEL_NAME"] = cleaned_model_name

    # Set up the extractor if not using pre-extracted definitions
    if args.extracted_definitions_path:
        logger.info(
            f"Using pre-extracted definitions from: {args.extracted_definitions_path}",
        )
        extractor = None
    else:
        API_BASE = "http://localhost:8000/v1"
        if args.load_compiled_path:
            # Setup DSPy
            lm = dspy.LM(
                args.llm_model_name,
                api_base=API_BASE,
                api_key="NONE",
                max_tokens=args.max_tokens,
            )
            dspy.configure(lm=lm)
            extractor = DSPyPaperExtractor(args.two_step_extraction)
            extractor.load(args.load_compiled_path)
        elif args.extractor_type == "PaperExtractor":
            extractor = DSPyPaperExtractor(args.two_step_extraction)
        else:
            llm_client = config.create_llm_client(
                model_name=cleaned_model_name,
                base_url=API_BASE,
            )
            if args.extractor_type == "MultiStepExtractor":
                extractor = ExtractionService(
                    extractor=MultiStepExtractor(llm_client=llm_client),
                    chunk_mode=ChunkMode(args.chunk_mode),
                )
            else:  # OneStepExtractor
                extractor = ExtractionService(
                    extractor=OnesStepExtractor(llm_client=llm_client),
                    chunk_mode=ChunkMode(args.chunk_mode),
                )

    # Load ground truth definitions and make splits
    gt = load_ground_truth(
        gt_definitions_path=args.ground_truth_path,
        grobid_extracted_papers_path=args.extractions_dir,
        chunk_mode=ChunkMode(args.chunk_mode),
    )
    trainset, devset, testset = make_splits(gt)
    logger.info(f"Train set size: {len(trainset)} examples")
    logger.info(f"Dev set size: {len(devset)} examples")
    logger.info(f"Test set size: {len(testset)} examples")
    logger.info(f"Total dataset size: {len(gt)} examples")

    # Run the actual DSPy extraction and evaluation
    wandb.init(
        project="definition_extraction",
        name=f"EVAL-{args.llm_model_name}_tau-{args.nli_threshold}_{'2S' if args.two_step_extraction else '1S'}",
        config={
            "llm": args.llm_model_name,
            "nli_model": args.nli_model_name,
            "nli_threshold": args.nli_threshold,
            "strict_vocab": not args.allow_out_of_vocab,
            "sections_mode": args.chunk_mode,
            "dataset_size": len(gt) if extractor else "N/A",
        },
    )

    # 1) PRE-DSPy evaluation
    if not args.skip_train:
        train_avg, train_rows = evaluate_and_log(
            program=extractor,
            dataset=trainset,  # evaluate on train
            tag="pre",
            nli_client=nli_client,
            tau=args.nli_threshold,
            strict_vocab=not args.allow_out_of_vocab,
            extracted_results_file=args.extracted_definitions_path
            if args.extracted_definitions_path
            else None,
        )
    if not args.skip_dev:
        dev_avg, dev_rows = evaluate_and_log(
            program=extractor,
            dataset=devset,  # evaluate on dev
            tag="pre",
            nli_client=nli_client,
            tau=args.nli_threshold,
            strict_vocab=not args.allow_out_of_vocab,
            extracted_results_file=args.extracted_definitions_path
            if args.extracted_definitions_path
            else None,
        )
    if not args.skip_test:
        test_avg, test_rows = evaluate_and_log(
            program=extractor,
            dataset=testset,  # evaluate on test
            tag="pre",
            nli_client=nli_client,
            tau=args.nli_threshold,
            strict_vocab=not args.allow_out_of_vocab,
            extracted_results_file=args.extracted_definitions_path
            if args.extracted_definitions_path
            else None,
        )

    # WandB logs
    if "train_avg" in locals() and not args.skip_train:
        wandb.log({"train_avg_score": train_avg})
    if "dev_avg" in locals() and not args.skip_dev:
        wandb.log({"dev_avg_score": dev_avg})
    if "test_avg" in locals() and not args.skip_test:
        wandb.log({"test_avg_score": test_avg})

    # Metrics to analyze in the detailed view
    stat_metrics = ["score", "num_sections", "num_predictions"]

    logger.info("Evaluation completed.\n Detailed Results:")

    results_buffer = []
    if "train_avg" in locals() and not args.skip_train:
        header = f"Train Average Score: {train_avg}"
        stats = get_detailed_stats(train_rows, stat_metrics)
        full_block = f"{header}\n{stats}\n"
        logger.info(full_block)
        results_buffer.append(full_block)

    if "dev_avg" in locals() and not args.skip_dev:
        header = f"Dev Average Score: {dev_avg}"
        stats = get_detailed_stats(dev_rows, stat_metrics)
        full_block = f"{header}\n{stats}\n"
        logger.info(full_block)
        results_buffer.append(full_block)

    if "test_avg" in locals() and not args.skip_test:
        header = f"Test Average Score: {test_avg}"
        stats = get_detailed_stats(test_rows, stat_metrics)
        full_block = f"{header}\n{stats}\n"
        logger.info(full_block)
        results_buffer.append(full_block)

    # Save the scores to a file with matching extracted_definitions_path if provided
    if args.extracted_definitions_path:
        scores_path = (
            args.extracted_definitions_path.parent
            / Path("scores")
            / (args.extracted_definitions_path.stem + "_scores.txt")
        )
        # Ensure directory exists
        scores_path.parent.mkdir(parents=True, exist_ok=True)

        with scores_path.open("w", encoding="utf-8") as f:
            for block in results_buffer:
                f.write(block + "\n")

        logger.info(f"Saved evaluation scores to: {scores_path}")


if __name__ == "__main__":
    main()
