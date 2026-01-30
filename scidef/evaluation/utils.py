# --- single-phase evaluation + logging ---
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import dspy

import wandb
from scidef.extraction.extractor.dspy_extraction import normalize_term
from scidef.extraction.service import ExtractionService
from scidef.model.dataclass import ExtractionResult
from scidef.model.nli.client import NLIClient
from scidef.model.nli.dataclass import ScoreMode
from scidef.utils import get_custom_colored_logger

logger = get_custom_colored_logger(__name__)


def _to_str_or_none(val) -> str | None:
    """Convert a value to string or None for wandb table compatibility."""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    if isinstance(val, (dict, list)):
        return str(val) if val else None
    return str(val)


def evaluate_and_log(
    program: Optional[Union[dspy.Module, ExtractionService]],
    dataset: List[dspy.Example],
    tag: str,
    nli_client: NLIClient,
    tau: float,
    strict_vocab: bool = False,
    extracted_results_file: Optional[Path] = None,
):
    assert program is not None or extracted_results_file is not None, (
        "Either extractor must be instantiated or extracted_results_file must be provided",
    )

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    preds_path = artifacts / f"preds_{tag}.jsonl"
    csv_path = artifacts / f"per_example_{tag}.csv"

    # Pre-load extracted definitions if using a file (avoid re-reading for each paper)
    extracted_definitions = None
    if extracted_results_file is not None:
        try:
            with extracted_results_file.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as fin:
                content = fin.read()
                # Remove any NUL bytes that might have slipped through
                content = content.replace("\x00", "")
                extracted_definitions = json.loads(content)["results"]
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse JSON from {extracted_results_file}: {e}. "
                f"The file may be corrupted or contain invalid characters.",
            )
            raise

    rows = []
    with preds_path.open("w", encoding="utf-8") as fout:
        for i, ex in enumerate(dataset):
            paper_id = getattr(ex, "paper_id", None) or "unknown"

            if program is not None:
                # Run extraction online
                if isinstance(program, ExtractionService):
                    pred = asyncio.run(
                        program.extract_definition(chunks=ex.sections),
                    )

                    merged = []
                    for d in pred["0"]:
                        if isinstance(d, ExtractionResult):
                            logger.warning(
                                f"Failed extraction result found in paper {paper_id}: {d}",
                            )
                            continue
                        if "error_message" in d:
                            logger.warning(
                                f"Failed extraction result found in paper {paper_id}: {d['error_message']}",
                            )
                            continue
                        # lowercase keys for consistency with DSPy predictions
                        merged.append(
                            {
                                "term": d["Term"]
                                if "Term" in d
                                else d["term"],
                                "definition": d["Definition"]
                                if "Definition" in d
                                else d["definition"],
                                "context": d.get("Context", "")
                                if "Context" in d
                                else d.get("context", ""),
                                "type": d.get("Type", "")
                                if "Type" in d
                                else d.get("type", ""),
                                "input": d.get("Input", "")
                                if "Input" in d
                                else d.get("input", ""),
                            },
                        )
                    logger.info(
                        f"Merged predictions for paper {paper_id}: {merged}",
                    )
                else:
                    pred = program(sections=ex.sections)
                    merged = json.loads(pred.merged_json)
            else:
                # Use pre-loaded extracted results
                assert extracted_definitions is not None
                base_id = paper_id.split("/")[-1]
                paper_keys = [
                    f"paper_{base_id}.grobid.tei",
                    f"{base_id}.grobid.tei",
                ]
                paper_results = None
                for key in paper_keys:
                    if key in extracted_definitions:
                        paper_results = extracted_definitions.get(key)
                        break
                if not paper_results:
                    logger.warning(
                        "No extracted definitions found for paper "
                        f"{paper_id} (keys tried: {paper_keys})",
                    )
                paper_results = list(paper_results or [])

                # lowercase keys for consistency with DSPy predictions
                pred = [
                    {
                        "term": d["Term"] if "Term" in d else d["term"],
                        "definition": d["Definition"]
                        if "Definition" in d
                        else d["definition"],
                        "context": d.get("Context", "")
                        if "Context" in d
                        else d.get("context", ""),
                        "type": d.get("Type", "")
                        if "Type" in d
                        else d.get("type", ""),
                        "input": d.get("Input", "")
                        if "Input" in d
                        else d.get("input", ""),
                    }
                    for d in paper_results
                    if "error_message" not in d
                ]
                merged = pred

            score = evaluate_extraction(
                example=ex,
                prediction=pred,  # type: ignore
                nli_client=nli_client,
                tau=tau,
                strict_vocab=strict_vocab,
                log_table=i % 24 == 0,  # log every 10th example's table
            )
            fout.write(
                json.dumps(
                    {
                        "paper_id": paper_id,
                        "predictions": merged,
                        "score": score,
                    },
                    ensure_ascii=False,
                )
                + "\n",
            )
            rows.append(
                {
                    "paper_id": paper_id,
                    "num_sections": len(ex.sections),
                    "num_predictions": len(merged),
                    "score": score,
                },
            )

    # rollups
    avg_nli = sum(r["score"] for r in rows) / len(rows) if rows else 0.0
    coverage_at_tau = (
        (sum(r["score"] >= tau for r in rows) / len(rows)) if rows else 0.0
    )
    avg_pairs = (
        sum(r["num_predictions"] for r in rows) / len(rows) if rows else 0.0
    )

    # log metrics (prefixed by phase tag)
    wandb.log(
        {
            f"{tag}_mean_nli": avg_nli,
            f"{tag}_coverage_at_tau": coverage_at_tau,
            f"{tag}_avg_pairs_per_paper": avg_pairs,
            f"{tag}_num_papers": len(rows),
        },
    )

    # log table
    import csv

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
            if rows
            else ["paper_id", "num_sections", "num_predictions", "score"],
        )
        w.writeheader()
        w.writerows(rows)

    wandb.log_artifact(str(preds_path))
    wandb.log_artifact(str(csv_path))
    return avg_nli, rows


def best_pred_for_gt(
    gt_term: Optional[str],
    gt_def: Optional[str],
    pred_pairs: List[Dict[str, str]],
    nli_client: NLIClient,
    tau: float = 0.60,
    strict_vocab: bool = False,
    gt_vocab_norm: Optional[set] = None,
) -> Tuple[Optional[Dict[str, str]], float]:
    if gt_def is None or gt_term is None:
        return None, 0.0
    best, best_score = None, -1.0
    for p in pred_pairs:
        if "term" not in p or "definition" not in p:
            continue
        # Skip entries with None/invalid terms or definitions
        if p["term"] is None or p["definition"] is None:
            continue
        if not isinstance(p["definition"], str):
            continue
        p_term = normalize_term(p["term"])
        if not p_term:  # Skip if term normalizes to empty
            continue
        if strict_vocab and gt_vocab_norm and p_term not in gt_vocab_norm:
            continue
        try:
            raw = nli_client.evaluate_bidirectional(
                gt_def,
                p["definition"],
                ScoreMode.AMEAN,
            )
            # Handle different response formats
            if "bidirectional_scores" in raw:
                score = float(raw["bidirectional_scores"][1])  # AMEAN score
            elif "score" in raw:
                score = float(raw["score"])
            else:
                continue
        except (KeyError, IndexError, TypeError):
            logger.warning(
                f"Skipping NLI evaluation for GT term '{gt_term}' and predicted term '{p['term']}' due to unexpected response format: {raw}",
            )
            continue
        if score > best_score:
            best, best_score = p, score
    if best is None or best_score < tau:
        return None, 0.0
    return best, best_score


def evaluate_extraction(
    example: dspy.Example,
    prediction: Union[dspy.Prediction, List[Dict[str, str]]],
    trace=None,
    nli_client: Optional[NLIClient] = None,
    tau: float = 0.60,
    strict_vocab: bool = True,
    log_table: bool = True,
) -> float:
    assert nli_client is not None, "NLI client is required"
    gt: dict = (
        example.ground_truth_definitions
    )  # {'term': {'definition': ..., 'context': ...}, ...}
    if not isinstance(prediction, dspy.Prediction):
        preds = prediction  # [{'term':..., 'definition':...}, ...]
    else:
        preds = json.loads(
            prediction.merged_json,
        )  # [{'term':..., 'definition':...}, ...]

    prediction_table = wandb.Table(
        columns=[
            "gt_term",
            "gt_definition",
            "gt_context",
            "gt_type",
            "pred_term",
            "pred_definition",
            "pred_context",
            "pred_type",
            "nli_score",
            "context_nli_score",
        ],
    )

    gt_vocab_norm = {normalize_term(t) for t in gt.keys()}
    gt_pred_cum_nli = 0.0

    logger.info("Ground Truth to Predicted Term Mapping:")
    for gt_term, gt_dict in gt.items():
        gt_def = gt_dict["definition"]
        gt_ctx = gt_dict["context"]
        gt_type = gt_dict["type"]

        best, score = best_pred_for_gt(
            gt_term,
            gt_def,
            preds,
            nli_client,
            tau=tau,
            strict_vocab=strict_vocab,
            gt_vocab_norm=gt_vocab_norm,
        )
        pred_term = _to_str_or_none(best["term"]) if best else None
        pred_def = _to_str_or_none(best["definition"]) if best else None
        pred_ctx = _to_str_or_none(best.get("context")) if best else None
        pred_type = _to_str_or_none(best.get("type")) if best else None

        # compute the context NLI score (WARNING: context might be long)
        ctx_score = 0.0
        if pred_ctx is not None:
            ctx_score = nli_client.evaluate_bidirectional(
                gt_ctx,
                pred_ctx,
                ScoreMode.AMEAN,
            )["bidirectional_scores"][1]

        type_score = 1.0 if (pred_type == gt_type) else 0.0

        logger.info(f"  Pred Term: {pred_term}  -->  GT Term: {gt_term}")
        logger.info(
            f"    Pred Def: {pred_def}\n    GT Def: {gt_def}\n\n    NLI Score: {score}\n",
        )
        logger.info(
            f"    Pred Context: {pred_ctx}\n    GT Context: {gt_ctx}\n",
        )
        logger.info(f"    Context NLI Score: {ctx_score}\n")
        logger.info(f"    GT Type: {gt_type}  Pred Type: {pred_type}\n")

        gt_pred_cum_nli += (score + ctx_score + type_score) / 3.0

        prediction_table.add_data(
            gt_term,
            gt_def,
            gt_ctx,
            gt_type,
            pred_term,
            pred_def,
            pred_ctx,
            pred_type,
            score,
            ctx_score,
        )

    pred_gt_cum_nli = 0.0
    logger.info("Predicted to Ground Truth Term Mapping:")
    for p in preds:
        if "term" not in p or "definition" not in p:
            continue
        pred_term = _to_str_or_none(p["term"])
        pred_def = _to_str_or_none(p["definition"])
        pred_ctx = _to_str_or_none(p.get("context", "")) or ""
        pred_type = _to_str_or_none(p.get("type", ""))

        best, score = best_pred_for_gt(
            pred_term,
            pred_def,
            [
                {
                    "term": t,
                    "definition": d["definition"],
                    "context": d["context"],
                    "type": d["type"],
                }
                for t, d in gt.items()
            ],
            nli_client,
            tau=tau,
            strict_vocab=strict_vocab,
            gt_vocab_norm=gt_vocab_norm,
        )
        gt_term = best["term"] if best else None
        gt_def = best["definition"] if best else None
        gt_ctx = best["context"] if best else None
        gt_type = best["type"] if best else None

        # compute the context NLI score (WARNING: context might be long)
        ctx_score = 0.0
        if gt_ctx is not None:
            ctx_score = nli_client.evaluate_bidirectional(
                pred_ctx,
                gt_ctx,
                ScoreMode.AMEAN,
            )["bidirectional_scores"][1]

        logger.info(f"  Pred Term: {pred_term}  -->  GT Term: {gt_term}")
        logger.info(
            f"    Pred Def: {pred_def}\n    GT Def: {gt_def}\n\n    NLI Score: {score}\n",
        )
        logger.info(
            f"    Pred Context: {pred_ctx}\n    GT Context: {gt_ctx}\n",
        )
        logger.info(f"    Context NLI Score: {ctx_score}\n")
        logger.info(f"    GT Type: {gt_type}  Pred Type: {pred_type}\n")

        type_score = 1.0 if (pred_type == gt_type) else 0.0
        pred_gt_cum_nli += (score + ctx_score + type_score) / 3.0

        prediction_table.add_data(
            gt_term,
            gt_def,
            gt_ctx,
            gt_type,
            pred_term,
            pred_def,
            pred_ctx,
            pred_type,
            score,
            ctx_score,
        )

    logger.info(
        f"Average NLI Score for this paper (GT to Pred): {gt_pred_cum_nli / len(gt)}\n{'-' * 40}\n All the GT terms were: {list(gt.keys())}\n All the Predicted terms were: {[p['term'] if 'term' in p else None for p in preds]}\n{'-' * 40}\n",
    )
    logger.info(
        f"Average NLI Score for this paper (Pred to GT): {pred_gt_cum_nli / len(preds) if preds else 0.0}\n{'-' * 40}\n",
    )

    logger.info(
        f"Final Average NLI Score for this paper:\n{(gt_pred_cum_nli + pred_gt_cum_nli) / (len(gt) + len(preds)) if preds else 0.0}\n{'-' * 40}\n",
    )

    final_score = (
        (gt_pred_cum_nli + pred_gt_cum_nli) / (len(gt) + len(preds))
        if preds
        else 0.0
    )

    wandb.log(
        {
            "example_gt_to_pred_avg_nli": gt_pred_cum_nli / len(gt),
            "example_pred_to_gt_avg_nli": pred_gt_cum_nli / len(preds)
            if preds
            else 0.0,
            "example_final_avg_nli": final_score,
            "example_prediction_table": prediction_table
            if log_table
            else None,
        },
    )

    return final_score
