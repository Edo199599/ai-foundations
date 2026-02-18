from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision_pos: float
    recall_pos: float
    f1_pos: float
    accuracy: float


def _binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    """Returns (tp, fp, fn, tn) for positive class = 1."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tp), int(fp), int(fn), int(tn)


def sweep_thresholds(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    thresholds: Optional[Iterable[float]] = None,
) -> List[ThresholdResult]:
    """
    Sweep thresholds for a binary classifier given y_proba_pos = P(y=1|x).
    Returns metrics for the positive class (1) at each threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba_pos = np.asarray(y_proba_pos).astype(float)

    if thresholds is None:
        thresholds = np.round(np.arange(0.1, 0.91, 0.1), 2)

    results: List[ThresholdResult] = []
    n = len(y_true)

    for thr in thresholds:
        y_pred = (y_proba_pos >= float(thr)).astype(int)

        tp, fp, fn, tn = _binary_confusion_counts(y_true, y_pred)

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[1],          # metrica solo per la classe positiva
            average=None,
            zero_division=0,     # evita warning quando non ci sono positivi predetti
        )

        accuracy = float((tp + tn) / n) if n > 0 else 0.0

        results.append(
            ThresholdResult(
                threshold=float(thr),
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision_pos=float(prec[0]),
                recall_pos=float(rec[0]),
                f1_pos=float(f1[0]),
                accuracy=accuracy,
            )
        )

    return results


def best_threshold_by_f1(results: List[ThresholdResult]) -> ThresholdResult:
    """Pick threshold that maximizes F1 for positive class. Tie-break: higher recall, then lower threshold."""
    if not results:
        raise ValueError("Empty results list.")
    return max(results, key=lambda r: (r.f1_pos, r.recall_pos, -r.threshold))


def print_threshold_table(results: List[ThresholdResult]) -> None:
    """Pretty-print a compact table without extra deps."""
    header = "thr |  TP  FP  FN  TN | prec1  rec1   f1_1 | acc"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.threshold:>3.2f} |"
            f"{r.tp:>4d}{r.fp:>4d}{r.fn:>4d}{r.tn:>4d} |"
            f"{r.precision_pos:>6.2f}{r.recall_pos:>6.2f}{r.f1_pos:>6.2f} |"
            f"{r.accuracy:>4.2f}"
        )

def best_threshold_with_min_recall(results: list[ThresholdResult], min_recall: float) -> ThresholdResult:
    """
    Among thresholds with recall_pos >= min_recall, pick the one with highest precision_pos,
    then highest f1, then lowest threshold.
    """
    eligible = [r for r in results if r.recall_pos >= min_recall]
    if not eligible:
        raise ValueError(f"No thresholds achieve recall >= {min_recall}.")
    return max(eligible, key=lambda r: (r.precision_pos, r.f1_pos, -r.threshold))