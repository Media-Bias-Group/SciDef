from enum import Enum
from typing import Any, Iterable, List


class MeasureMethod(Enum):
    """Different prompts for LLM judge similarity evaluation."""

    PEARSON = "pearson"
    F1 = "f1"
    ACCURACY = "accuracy"


def get_judgment_index_from_score(
    similarity_score: float,
    bucket_order: List[Any],
) -> int:
    if not 0.0 <= similarity_score <= 1.0:
        raise ValueError("Similarity score must be between 0 and 1.")

    bucket_size = 1.0 / len(bucket_order)
    index = min(int(similarity_score / bucket_size), len(bucket_order) - 1)
    return index


def measure_performance(
    pred: Iterable[Any],
    gt: Iterable[Any],
    method: MeasureMethod = MeasureMethod.PEARSON,
):
    if method == MeasureMethod.PEARSON:
        return _compute_pearson(pred=pred, gt=gt)
    elif method == MeasureMethod.F1:
        return _compute_f1_score(pred=pred, gt=gt)
    elif method == MeasureMethod.ACCURACY:
        return _compute_accuracy_score(pred=pred, gt=gt)
    else:
        raise Exception(f"Method {method} not implemented.")


def _compute_pearson(pred: Iterable[Any], gt: Iterable[Any]):
    from scipy.stats import pearsonr

    return float(pearsonr(pred, gt).statistic)


def _compute_f1_score(pred: Iterable[Any], gt: Iterable[Any]):
    from sklearn.metrics import f1_score

    return f1_score(y_pred=pred, y_true=gt, average="macro")


def _compute_accuracy_score(pred: Iterable[Any], gt: Iterable[Any]):
    from sklearn.metrics import accuracy_score

    return accuracy_score(y_pred=pred, y_true=gt)
