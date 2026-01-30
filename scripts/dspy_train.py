import argparse
import logging
import time
from functools import partial
from pathlib import Path
from typing import Literal

import dspy

import wandb
from _bootstrap import load_config_module
from scidef.evaluation.utils import evaluate_and_log, evaluate_extraction
from scidef.extraction.dataclass import ChunkMode
from scidef.extraction.extractor.dspy_extraction import DSPyPaperExtractor
from scidef.extraction.utils import load_ground_truth, make_splits
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)
_config = load_config_module()
Config = _config.Config
setup_logging = _config.setup_logging


def optimize_and_save(
    program,
    trainset,
    devset,
    nli_client,
    tau,
    strict_vocab,
    metric_threshold: float,
    save_path="artifacts/dspy_paper_extractor_v1.json",
    num_mipro_threads=4,
    optimizer: Literal["miprov2", "bootstrap", "bootstrap_random"] = "miprov2",
    mipro_auto: Literal["light", "medium", "heavy"] = "light",
):
    opt_params = {
        "mipro_max_bootstrapped_demos": 4,
        "mipro_init_temperature": 0.7,
        "mipro_auto": mipro_auto,
        "metric_threshold": metric_threshold,
    }
    wandb.log(opt_params)

    if optimizer == "miprov2":
        optim = dspy.MIPROv2(
            metric=partial(
                evaluate_extraction,
                nli_client=nli_client,
                tau=tau,
                strict_vocab=strict_vocab,
            ),
            num_threads=num_mipro_threads,
            max_bootstrapped_demos=int(
                opt_params["mipro_max_bootstrapped_demos"],
            ),
            init_temperature=float(opt_params["mipro_init_temperature"]),
            auto=mipro_auto,
        )
    elif optimizer == "bootstrap":
        optim = dspy.BootstrapFewShot(
            metric=partial(
                evaluate_extraction,
                nli_client=nli_client,
                tau=tau,
                strict_vocab=strict_vocab,
            ),
            max_bootstrapped_demos=int(
                opt_params["mipro_max_bootstrapped_demos"],
            ),
        )
    elif optimizer == "bootstrap_random":
        optim = dspy.BootstrapFewShotWithRandomSearch(
            metric=partial(
                evaluate_extraction,
                nli_client=nli_client,
                tau=tau,
                strict_vocab=strict_vocab,
            ),
            max_bootstrapped_demos=int(
                opt_params["mipro_max_bootstrapped_demos"],
            ),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    if isinstance(optim, dspy.BootstrapFewShot):
        compiled = optim.compile(program, trainset=trainset)
    elif isinstance(
        optim,
        dspy.BootstrapFewShotWithRandomSearch,
    ) or isinstance(optim, dspy.MIPROv2):
        compiled = optim.compile(
            program,
            trainset=trainset,
            valset=devset,
        )
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    compiled.save(save_path)
    wandb.save(save_path)

    return compiled


def main():
    parser = argparse.ArgumentParser(
        description="Extract definitions from academic papers",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        help="path containing ground truths",
        default="data/defExtra.json",
    )
    parser.add_argument(
        "--extractions-dir",
        type=Path,
        nargs="+",
        default=[
            Path("ManualPDFsGROBID/manual_pdfs_grobid"),
            Path("ManualPDFsGROBID/new_grobid"),
        ],
        help="Extractions directory",
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
        default=0.25,
        help="NLI score threshold for correctness",
    )

    parser.add_argument(
        "--allow-out-of-vocab",
        action="store_true",
        help="Allow out-of-vocabulary terms during evaluation",
    )

    parser.add_argument(
        "--num-mipro-threads",
        type=int,
        default=4,
        help="Number of threads (concurrent requests) for MIPROv2 (DSPy) optimization",
    )

    parser.add_argument(
        "--eval-pre-dspy",
        action="store_true",
        help="Evaluate pre-DSPy extraction performance",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for LLM responses",
    )

    parser.add_argument(
        "--dspy-optimizer",
        type=str,
        choices=["miprov2", "bootstrap", "bootstrap_random"],
        default="miprov2",
        help="DSPy optimizer to use"
        "(miprov2: MIPROv2, bootstrap: BootstrapFewShot, bootstrap_random: BootstrapFewShotWithRandomSearch) - default miprov2 but requires ~200+ examples.",
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
        "--nli-compile",
        action="store_true",
        help="Compile NLI model for faster inference",
    )

    parser.add_argument(
        "--metric-threshold",
        type=float,
        default=0.25,
        help="Metric threshold for DSPy optimization",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable DSPy caching (for debugging purposes)",
    )

    parser.add_argument(
        "--base-api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base API URL for LLM requests",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="NONE",
        help="API key for LLM requests",
    )

    parser.add_argument(
        "--mipro-auto",
        type=str,
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 auto setting (light, medium, heavy)",
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
        f"Number of MIPROv2 threads: {args.num_mipro_threads}",
    )
    logger.info(
        f"Evaluate pre-DSPy performance: {args.eval_pre_dspy}",
    )
    logger.info(
        f"Use two-step extraction process: {args.two_step_extraction}",
    )
    logger.info(f"DSPy optimizer: {args.dspy_optimizer}")
    logger.info(f"Chunk mode: {args.chunk_mode}")
    logger.info(f"NLI compile: {args.nli_compile}")
    logger.info(f"Metric threshold: {args.metric_threshold}")
    logger.info(f"Disable DSPy cache: {args.disable_cache}")
    logger.info(f"Base API URL: {args.base_api_url}")
    logger.info(f"API Key: {'SET' if args.api_key != 'NONE' else 'NOT SET'}")
    logger.info(f"MIPRO auto setting: {args.mipro_auto}")

    lm = dspy.LM(
        args.llm_model_name,
        api_base=args.base_api_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        cache=not args.disable_cache,
    )
    dspy.configure(lm=lm)

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

    gt = load_ground_truth(
        gt_definitions_path=str(args.ground_truth_path),
        grobid_extracted_papers_path=args.extractions_dir,
        chunk_mode=ChunkMode(args.chunk_mode),
    )
    trainset, devset, testset = make_splits(gt)
    trainset = list(trainset)
    devset = list(devset)
    testset = list(testset)

    logger.info(f"Train set size: {len(trainset)} examples")
    logger.info(f"Dev set size: {len(devset)} examples")
    logger.info(f"Test set size: {len(testset)} examples")
    logger.info(f"Total dataset size: {len(gt)} examples")

    wandb.init(
        project="definition_extraction",
        name=f"{args.llm_model_name}_tau-{args.nli_threshold}_{'2S' if args.two_step_extraction else '1S'}",
        config={
            "llm": args.llm_model_name,
            "nli_model": args.nli_model_name,
            "nli_threshold": args.nli_threshold,
            "strict_vocab": not args.allow_out_of_vocab,
            "sections_mode": args.chunk_mode,
            "dataset_size": len(gt),
        },
    )
    logging.getLogger("weave.trace.weave_client").setLevel(logging.ERROR)

    # 1) PRE-DSPy evaluation
    if args.eval_pre_dspy:
        pre_avg, _ = evaluate_and_log(
            program=DSPyPaperExtractor(
                args.two_step_extraction,
            ),
            dataset=devset,  # evaluate on dev
            tag="pre",
            nli_client=nli_client,
            tau=args.nli_threshold,
            strict_vocab=not args.allow_out_of_vocab,
        )

        wandb.log({"pre_dspy_avg_f1": pre_avg})

    # 2) Optimize with MIPROv2 (train on trainset, validate on devset), save compiled state

    _ = optimize_and_save(
        program=DSPyPaperExtractor(args.two_step_extraction),
        trainset=trainset,
        devset=devset,
        nli_client=nli_client,
        tau=args.nli_threshold,
        strict_vocab=not args.allow_out_of_vocab,
        save_path=f"artifacts/dspy_paper_extractor_{args.llm_model_name}_tau-{args.nli_threshold}_metricThres{args.metric_threshold}_{args.chunk_mode}_{args.dspy_optimizer}-{args.mipro_auto}_{time.time()}_v1.json",
        num_mipro_threads=args.num_mipro_threads,
        optimizer=args.dspy_optimizer,
        metric_threshold=args.metric_threshold,
        mipro_auto=args.mipro_auto,
    )

    logger.info("Optimization complete and compiled program saved!")


if __name__ == "__main__":
    main()
