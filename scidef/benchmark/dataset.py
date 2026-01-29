import csv
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import kagglehub
import numpy as np
import tiktoken
from datasets import load_dataset

from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


def estimate_token_use(pairs: List[Tuple[str, str, float]]) -> int:
    """Estimate token use of a list of string pairs"""
    enc = tiktoken.get_encoding(
        "o200k_base",
    )  # the openai default, should approximate other models as well
    estimated_tokens = 0
    for sent1, sent2, _ in pairs:
        estimated_tokens += len(enc.encode(sent1) + enc.encode(sent2))
    return estimated_tokens


def load_sts3k(
    split: str = "sts3k-all",
    sample_size: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Load STS3k dataset.
    The dataset is downloaded from:
    https://github.com/bmmlab/compositional-semantics-eval/
    """
    part = split.split("-")[-1]
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        local_path_all = data_dir / "STS3k_all.txt"
        local_path_non_adv_ind = data_dir / "STS3k_non_adv_indices.txt"

        if not local_path_all.exists():
            url = "https://raw.githubusercontent.com/bmmlab/compositional-semantics-eval/master/Data-experiment/STS3k_all.txt"
            logger.info(f"Downloading STS3k-all dataset from {url}")
            _ = urllib.request.urlretrieve(url, local_path_all)
            logger.info(f"Downloaded STS3k-all dataset to {local_path_all}")
        if "all" not in split:
            if not local_path_non_adv_ind.exists():
                url = "https://raw.githubusercontent.com/bmmlab/compositional-semantics-eval/master/Data-experiment/STS3k_non_adv_indices.txt"
                logger.info(f"Downloading STS3k-non_adv indices from {url}")
                _ = urllib.request.urlretrieve(url, local_path_non_adv_ind)
                logger.info(
                    f"Downloaded STS3k-non_adv indices to {local_path_all}",
                )

        # Load and parse the data
        pairs = []
        with open(local_path_all, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if len(row) == 3:
                    try:
                        pairs.append((row[0], row[1], float(row[2])))
                    except ValueError:
                        logger.warning(f"Error while reading row {row}")
                        continue

        if split != "sts3k-all":
            non_adv_indices = np.loadtxt(
                local_path_non_adv_ind,
                delimiter=",",
                dtype="int",
                encoding="utf-8",
                skiprows=0,
            )
            pairs = np.asarray(pairs)
            if split == "sts3k-non":
                pairs = pairs[non_adv_indices]
            elif split == "sts3k-adv":
                pairs = pairs[~non_adv_indices]
            else:
                logger.error(f"Unknown split '{split}' passed to load_sts3k.")
                return []
            pairs = [(p[0], p[1], float(p[2])) for p in pairs]

        if sample_size:
            pairs = pairs[:sample_size]

        logger.info(f"Loaded {len(pairs)} STS3k pairs")
        return pairs

    except Exception as e:
        logger.error(f"Failed to load STS3k-{part}: {e}")
        return []


def load_msr_paraphrases(
    split: str = "train",
    sample_size: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Load Microsoft Research Paraphrase Corpus
    https://www.kaggle.com/datasets/doctri/microsoft-research-paraphrase-corpus
    """

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    local_path_all = data_dir / f"msr_paraphrase_{split}.txt"
    try:
        with open(local_path_all, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            pairs = []
            for row in reader:
                if len(row) == 5:
                    try:
                        pairs.append((row[3], row[4], float(row[0])))
                    except ValueError:
                        logger.warning(f"Error while reading row {row}")
                        continue
        if sample_size:
            pairs = pairs[:sample_size]
        logger.info(f"Loaded {len(pairs)} MSR Paraphrase pairs")
        return pairs

    except Exception as e:
        logger.error(f"Failed to load MSR Paraphrases: {e}")
        return []


def load_quora_duplicates(
    split: str = "train",
    sample_size: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Load Question Pairs Dataset"""
    try:
        ds_path = kagglehub.dataset_download("quora/question-pairs-dataset")
        file_path = os.path.join(ds_path, "questions.csv")
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            pairs = []
            for row in reader:
                if row[0] == "id":
                    continue  # Skip header
                if len(row) == 6:
                    try:
                        pairs.append((row[3], row[4], float(row[5])))
                    except ValueError:
                        logger.warning(f"Error while reading row {row}")
                        continue

        if sample_size:
            pairs = pairs[:sample_size]

        logger.info(f"Loaded {len(pairs)} Question Pairs Dataset pairs")
        return pairs
    except Exception as e:
        logger.error(f"Failed to load Question Pairs Dataset: {e}")
        return []


def load_stsb(
    split: str = "train",
    sample_size: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Load STS-B dataset."""
    try:
        dataset = load_dataset("sentence-transformers/stsb", split=split)
        pairs = [
            (item["sentence1"], item["sentence2"], float(item["score"]))
            for item in dataset
        ]

        if sample_size:
            pairs = pairs[:sample_size]

        logger.info(f"Loaded {len(pairs)} STS-B pairs")
        return pairs
    except Exception as e:
        logger.error(f"Failed to load STS-B: {e}")
        return []


def load_sick(
    split: str = "train",
    sample_size: Optional[int] = None,
) -> List[Tuple[str, str, float]]:
    """Load SICK dataset.

    Downloads and parses the SICK dataset similarly to the deprecated
    HuggingFace datasets script. Data is fetched from Zenodo, extracted,
    and filtered by split.
    """
    try:
        # Set up local data directory
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)

        # Download archive if needed
        download_url = (
            "https://zenodo.org/record/2787612/files/SICK.zip?download=1"
        )
        zip_path = data_dir / "SICK.zip"
        extract_dir = data_dir / "SICK"

        if not extract_dir.exists() or not (extract_dir / "SICK.txt").exists():
            if not zip_path.exists():
                logger.info(f"Downloading SICK dataset from {download_url}")
                _ = urllib.request.urlretrieve(download_url, zip_path)
                logger.info(f"Downloaded SICK dataset to {zip_path}")

            # Extract archive
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)

        # Determine file path
        file_path = extract_dir / "SICK.txt"
        if not file_path.exists():
            # Fallback: handle unexpected extraction layout
            alt_path = data_dir / "SICK.txt"
            if alt_path.exists():
                file_path = alt_path
            else:
                raise FileNotFoundError("SICK.txt not found after extraction")

        # Map split to file key
        split_key = {
            "train": "TRAIN",
            "training": "TRAIN",
            "validation": "TRIAL",
            "valid": "TRIAL",
            "val": "TRIAL",
            "trial": "TRIAL",
            "dev": "TRIAL",
            "test": "TEST",
        }.get(split.lower(), None)

        if split_key is None:
            logger.warning(f"Unknown split '{split}', defaulting to 'TRAIN'")
            split_key = "TRAIN"

        # Parse and filter rows
        pairs: List[Tuple[str, str, float]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # SICK.txt is tab-separated; last field is the split key
                fields = [s.strip() for s in line.split("\t")]
                if not fields:
                    continue
                if fields[-1] != split_key:
                    continue

                # Expected columns as per HF script
                # 0: id, 1: sentence_A, 2: sentence_B, 4: relatedness_score
                try:
                    sentence_a = fields[1]
                    sentence_b = fields[2]
                    relatedness = float(fields[4]) / 5.0
                except Exception:
                    # Skip malformed rows and possible headers
                    continue

                pairs.append((sentence_a, sentence_b, relatedness))

        if sample_size:
            pairs = pairs[:sample_size]

        logger.info(f"Loaded {len(pairs)} SICK pairs for split '{split}'")
        return pairs
    except Exception as e:
        logger.error(f"Failed to load SICK: {e}")
        return []
