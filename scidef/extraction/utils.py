import json
from pathlib import Path
from typing import List

import dspy
import torch
from torch.utils.data import random_split
from tqdm import tqdm

from scidef.extraction.dataclass import ChunkMode
from scidef.grobid.service import extract_text_from_grobid
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


# --- splits (TRAIN/DEV/TEST) ---
def make_splits(
    dataset,
    train_frac: float = 0.6,
    dev_frac: float = 0.2,
    test_frac: float = 0.2,
):
    n = len(dataset)
    n_train = int(train_frac * n)
    n_dev = int(dev_frac * n)
    n_test = n - n_train - n_dev

    torch.manual_seed(42)  # for reproducibility

    splits = random_split(dataset, [n_train, n_dev, n_test])

    return splits


def load_ground_truth(
    gt_definitions_path: str,
    grobid_extracted_papers_path: List[Path],
    chunk_mode: ChunkMode,
) -> List[dspy.Example]:
    """Load ground truth definitions from a JSON file.

    Args:
        gt_definitions_path (str): Path to the ground truth definitions JSON file.
        grobid_extracted_papers_path (Path): Path to the directory containing GROBID extracted papers.
        chunk_mode (ChunkMode): The chunking mode to use when extracting text from GROBID files.

    Returns:
        Dict[str, Dict[str, Dict[str, str]]]: A dictionary mapping paper IDs to their ground truth definitions & contexts.
    """
    ground_truth = None
    with open(gt_definitions_path, "r") as f:
        ground_truth = json.load(f)

    # Load the paper contents for the papers for which we have extractions in the ground truth
    extractions = []
    for grobid_dir in grobid_extracted_papers_path:
        assert grobid_dir.exists(), (
            f"GROBID extracted papers path does not exist: {grobid_dir}"
        )

        for gt_file in tqdm(
            ground_truth.keys(),
            desc=f"Loading GROBID extractions from {grobid_dir}",
        ):
            grobid_file = (
                grobid_dir / f"paper_{gt_file}.grobid.tei.xml"
                if "/" not in gt_file
                else grobid_dir
                / f"paper_{gt_file.split('/')[-1]}.grobid.tei.xml"
            )

            if not grobid_file.exists():
                continue

            logger.debug(f"Loading extraction file: {grobid_file}")
            data = extract_text_from_grobid(grobid_file, chunk_mode=chunk_mode)
            extractions.append(
                {
                    "file": gt_file,
                    "data": data,
                },
            )

    # Print papers for which we have a GT but we didn't find them
    for gt_paper_id in ground_truth.keys():
        if not any(ext["file"] == gt_paper_id for ext in extractions):
            logger.warning(
                f"Ground truth paper {gt_paper_id} has no corresponding extraction file in {grobid_extracted_papers_path}",
            )

    logger.info(f"Loaded {len(extractions)} extraction files")

    dataset = []
    for extraction in tqdm(extractions):
        name = extraction["file"]  # paper_{paper_id}.tei.xml
        text_chunks = extraction["data"]  # List of text chunks

        assert len(text_chunks) > 0, (
            f"No text chunks found for paper {name} in GROBID extractions",
        )

        gt_definitions = ground_truth.get(
            name,
            dict(),
        )  # dict like "{'gender bias': {'definition': 'Not equal representation of male in female in various contexts.', 'context': '...', 'type': 'explicit'}, ...}"
        if not gt_definitions:
            logger.warning(
                f"No ground truth definitions found for paper {name}",
            )
            continue

        dataset.append(
            dspy.Example(
                paper_id=name,
                sections=text_chunks,
                ground_truth_definitions=gt_definitions,
            ).with_inputs("sections"),
        )
    logger.info(f"Prepared dataset with {len(dataset)} examples")
    return dataset
