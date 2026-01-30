import argparse
import asyncio
import json
import os
import pathlib
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

import dspy

from _bootstrap import load_config_module
from scidef.extraction.dataclass import ChunkMode
from scidef.extraction.extractor import (
    MultiStepExtractor,
    MultiStepFewShotExtractor,
    OnesStepExtractor,
    OnesStepFewShotExtractor,
)
from scidef.extraction.extractor.dspy_extraction import DSPyPaperExtractor
from scidef.extraction.service import ExtractionService
from scidef.grobid.service import extract_metadata_from_grobid
from scidef.model.dataclass import ExtractionResult, PaperMetadata
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)
Config = load_config_module().Config


EXTRACTOR_CLASSES = {
    "DSPyPaperExtractor": DSPyPaperExtractor,
    "MultiStepExtractor": MultiStepExtractor,
    "OneStepExtractor": OnesStepExtractor,
    "OneStepFewShotExtractor": OnesStepFewShotExtractor,
    "MultiStepFewShotExtractor": MultiStepFewShotExtractor,
}


def candidate_grobid_paths(grobid_dir: Path, paper_id: str) -> list[Path]:
    """Return GROBID TEI candidate paths for a given paper id."""
    base = paper_id.split("/")[-1]
    return [
        grobid_dir / f"paper_{base}.grobid.tei.xml",
        grobid_dir / f"{base}.grobid.tei.xml",
    ]


def sanitize_string(s: str) -> str:
    """Remove or replace invalid Unicode characters from a string for safe JSON serialization."""
    result = []
    for c in s:
        code = ord(c)
        # - Surrogate characters (U+D800 to U+DFFF) - invalid in UTF-8
        # - NUL byte (U+0000) - can cause issues in some parsers
        # - Other control chars that might cause issues (but keep common ones like \n, \t, \r)
        if 0xD800 <= code <= 0xDFFF:
            # Replace surrogates with replacement character
            result.append("\ufffd")
        elif code == 0:
            # Skip NUL bytes entirely
            continue
        else:
            result.append(c)
    return "".join(result)


def serialize_for_json(obj: Any) -> Any:
    """Recursively convert dataclasses and enums to JSON-serializable types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: serialize_for_json(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, str):
        return sanitize_string(obj)
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            serialize_for_json(k): serialize_for_json(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, set):
        return [serialize_for_json(item) for item in obj]
    return obj


def save_results(
    results: dict[str, list | ExtractionResult],
    output_dir: Path,
    extractor_name: str,
    chunk_mode: str,
    num_papers: int,
    model: str,
    prompts_or_dspy_program: str,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
) -> Path:
    """Save results to a JSON file with metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = [
        "results",
        model.replace("/", "_"),
        extractor_name,
        chunk_mode,
        f"{num_papers}_papers",
        timestamp,
    ]

    # Add LLM parameters if specified
    if temperature is not None:
        parts.append(f"temp{temperature}")
    if top_p is not None:
        parts.append(f"topp{top_p}")

    filename = "_".join(parts) + ".json"

    output_path = output_dir / filename
    output_data = {
        "metadata": {
            "extractor": extractor_name,
            "chunk_mode": chunk_mode,
            "model": model,
            "num_papers": num_papers,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "temperature": temperature,
                "top_p": top_p,
            },
            "prompts_or_dspy_program": str(prompts_or_dspy_program),
        },
        "results": serialize_for_json(results),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {output_path}")
    return output_path


async def main():
    parser = argparse.ArgumentParser(
        description="Extract definitions from academic papers",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        nargs="+",
        help="Directory containing XML files",
    )

    parser.add_argument(
        "--gt_path",
        type=str,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory",
        default="results/extraction",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Maximum number of papers to process",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for LLM extraction (0.0 to 2.0, only used if specified)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p for LLM extraction (0.0 to 1.0, only used if specified)",
    )

    parser.add_argument(
        "--chunk_modes",
        nargs="+",
        choices=[m.value for m in ChunkMode],
        default=[
            ChunkMode.THREE_SENTENCE.value,
            ChunkMode.SECTION.value,
            ChunkMode.PARAGRAPH.value,
            ChunkMode.SENTENCE.value,
        ],
        help="Processing modes",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=list(EXTRACTOR_CLASSES.keys()),
        default=[
            "MultiStepExtractor",
            "OneStepExtractor",
            "OneStepFewShotExtractor",
            "MultiStepFewShotExtractor",
        ],  # Should be a list since nargs="+"
        help="Extractor modes to use",
    )

    parser.add_argument(
        "--llm-model-name",
        type=str,
        help="LLM model name for DSPy",
    )

    parser.add_argument(
        "--two-step-extraction",
        action="store_true",
        help="Enable two-step extraction for DSPyPaperExtractor",
    )

    parser.add_argument(
        "--dspy-program-path",
        type=str,
        default="",
        help="Path to DSPy program file (if using DSPyPaperExtractor)",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for LLM responses",
    )

    parser.add_argument(
        "--max-concurrent-papers",
        type=int,
        default=64,
        help="Maximum number of concurrent papers to process",
    )

    parser.add_argument(
        "--base-api-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for LLM API",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default="NONE",
        help="API key for LLM service",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for LLM response caching",
    )

    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Disable LLM response caching",
    )

    args = parser.parse_args()

    config = Config()
    os.environ["LLM_MODEL_NAME"] = args.llm_model_name
    os.environ["VLLM_MODEL_NAME"] = args.llm_model_name
    os.environ["OPENROUTER_MODEL_NAME"] = args.llm_model_name
    logger.setLevel(args.log_level)

    xml_files = []

    if args.gt_path:
        with open(args.gt_path, "r") as f:
            ground_truth = json.load(f)
            for grobid_dir in args.input_dir:
                assert grobid_dir.exists(), (
                    f"GROBID extracted papers path does not exist: {grobid_dir}"
                )
                for gt_file in ground_truth.keys():
                    for candidate in candidate_grobid_paths(
                        grobid_dir,
                        gt_file,
                    ):
                        if candidate.exists():
                            xml_files.append(candidate)
                            break
    else:
        for input_dir in args.input_dir:
            input_dir = pathlib.Path(input_dir)
            xml_files.extend(input_dir.glob("*.grobid.tei.xml"))

    if not xml_files:
        logger.error(f"No XML files found in {args.input_dir}")
        return

    logger.info(
        f"Found {len(xml_files)} XML files to process from {len(args.input_dir)} directories.",
    )
    if args.temperature is not None:
        logger.info(f"LLM temperature: {args.temperature}")
    if args.top_p is not None:
        logger.info(f"LLM top-p: {args.top_p}")

    extractors = []
    for name in args.extractors:
        if name not in EXTRACTOR_CLASSES:
            logger.error(f"Unknown extractor: {name}. Skipping...")
            continue
        extractor_class = EXTRACTOR_CLASSES.get(name)
        if extractor_class is None:
            logger.error(f"Extractor class for {name} not found. Skipping...")
            continue
        if extractor_class is DSPyPaperExtractor:
            lm = dspy.LM(
                args.llm_model_name,
                api_base=args.base_api_url,
                api_key=args.api_key,
                max_tokens=args.max_tokens,
                cache=True if args.max_concurrent_papers <= 32 else False,
            )
            dspy.configure(lm=lm)
            extractor = extractor_class(two_step=args.two_step_extraction)
            if args.dspy_program_path:
                extractor.load(args.dspy_program_path)
        else:
            extractor = extractor_class(
                config.create_llm_client(
                    model_name=args.llm_model_name,
                    base_url=args.base_api_url,
                    cache_dir=args.cache_dir,
                    disable_cache=args.disable_cache,
                    api_key=args.api_key if args.api_key != "NONE" else None,
                    log_level=args.log_level,
                ),
                temperature=args.temperature or None,
                top_p=args.top_p or None,
            )
        extractors.append(extractor)

    metadatas: List[PaperMetadata] = []
    for i, xml_file in enumerate(xml_files, 1):
        if args.max_papers and i > args.max_papers:
            break

        try:
            paper_id = xml_file.stem
            logger.info(f"Processing paper {i}/{len(xml_files)}: {paper_id}")

            # Create paper metadata
            metadata = PaperMetadata(
                paper_id=paper_id,
                xml_file_path=xml_file,
                **extract_metadata_from_grobid(xml_file),
            )
            metadatas.append(metadata)

        except Exception as e:
            logger.error(f"Error processing {xml_file}: {e}")
            continue

    sem = asyncio.Semaphore(args.max_concurrent_papers)

    async def run_extraction(extractor, chunk_mode):
        logger.info(
            f"Running {extractor.__class__.__name__} with chunk_mode={chunk_mode}",
        )

        extraction_service = ExtractionService(
            extractor=extractor,
            chunk_mode=ChunkMode(chunk_mode),
            max_concurrent_papers=args.max_concurrent_papers,
        )

        results = await extraction_service.extract_definitions(
            paper_metadatas=metadatas,
            sem=sem,
        )
        return extractor, chunk_mode, results

    tasks = [
        run_extraction(extractor, chunk_mode)
        for chunk_mode in args.chunk_modes
        for extractor in extractors
    ]

    for extractor, chunk_mode, results in await asyncio.gather(*tasks):
        prompts_or_dspy_program = ""
        if isinstance(extractor, DSPyPaperExtractor):
            prompts_or_dspy_program = (
                args.dspy_program_path
                if args.dspy_program_path
                else "No DSPy program loaded"
            )
        elif isinstance(
            extractor,
            (MultiStepExtractor, MultiStepFewShotExtractor),
        ):
            from scidef.extraction.extractor.multi_step_extraction import (
                create_binary_llm,
                create_extraction_llm,
            )

            prompts_or_dspy_program = {
                "binary_system": create_binary_llm("")[0],
                "binary_instruction": create_binary_llm("")[1],
                "extraction_system": create_extraction_llm("")[0],
                "extraction_instruction": create_extraction_llm("")[1],
            }
        elif isinstance(
            extractor,
            (OnesStepExtractor, OnesStepFewShotExtractor),
        ):
            from scidef.extraction.extractor.one_step_extraction import (
                indentify_extract_llm,
            )

            prompts_or_dspy_program = {
                "system": indentify_extract_llm("")[0],
                "instruction": indentify_extract_llm("")[1],
            }

        save_results(
            results=results,
            output_dir=args.output_dir,
            extractor_name=extractor.__class__.__name__,
            chunk_mode=chunk_mode,
            num_papers=len(metadatas),
            temperature=args.temperature,
            top_p=args.top_p,
            model=args.llm_model_name,
            prompts_or_dspy_program=str(prompts_or_dspy_program),
        )


if __name__ == "__main__":
    asyncio.run(main())
