from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, List

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import math


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
    support_pos: int
    accuracy: float
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    mcc: float = 0.0


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

        prec, rec, f1, sup = precision_recall_fscore_support(
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
                support_pos=int(sup[0]),
                accuracy=accuracy
            )
        )

    return results

def evaluate_standard(y_true: np.ndarray, y_pred: np.ndarray) -> ThresholdResult:
    """
    Standard evaluation for hard predictions (0/1).
    Returns a ThresholdResult with threshold set to NaN (not applicable here).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred must have same length: {len(y_true)} != {len(y_pred)}")

    tp, fp, fn, tn = _binary_confusion_counts(y_true, y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1], # specifico di volere le metriche della classe 1 come positiva
        average=None,
        zero_division=0
    )
    precision_pos = float(prec[0])
    recall_pos = float(rec[0])
    f1_pos = float(f1[0])
    support_pos = int(sup[0])

    n = len(y_true)
    accuracy = float((tp + tn)/ n) if n > 0 else 0.0

    specificity = (tn / (tn + fp)) if tn + fp != 0 else 0.0 # recall classe 0
    balanced_accuracy = (recall_pos + specificity) / 2.0 # media recall delle due classi

    denom = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    mcc = (tp*tn - fp*fn) / math.sqrt(denom) if denom != 0 else 0.0

    return ThresholdResult(
        threshold=float("nan"),
        tp = tp,
        fp = fp,
        fn = fn,
        tn = tn,
        precision_pos = precision_pos,
        recall_pos = recall_pos,
        f1_pos = f1_pos,
        support_pos = support_pos,
        accuracy = accuracy,
        specificity = float(specificity),
        balanced_accuracy = float(balanced_accuracy),
        mcc = float(mcc)
    )

def print_standard_eval(res: ThresholdResult, *, title: str | None = None) -> None:
    if title:
        print("\n" + title)
    print(f"TP={res.tp} FP={res.fp} FN={res.fn} TN={res.tn}")
    print(
        f"precision={res.precision_pos:.4f} | recall={res.recall_pos:.4f} | f1={res.f1_pos:.4f} | acc={res.accuracy:.4f} | sup={res.support_pos}"
    )
    print(
        f"specificity(TNR)={res.specificity:.4f} | balanced_acc={res.balanced_accuracy:.4f} | MCC={res.mcc:.4f}"
    )

def best_threshold_by_f1(results: List[ThresholdResult]) -> ThresholdResult:
    """
    Select the threshold that maximizes F1 for the positive class (label=1).

    Tie-break policy (in order):
      1) higher recall_pos (prefer fewer false negatives)
      2) lower threshold (less conservative) via -threshold in the key

    Rationale:
      - If two thresholds yield the same F1, we bias towards catching positives (recall).
      - If still tied, we prefer the smaller threshold which typically predicts positives more often.
    """
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
