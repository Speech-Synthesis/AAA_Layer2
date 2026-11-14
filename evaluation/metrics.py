"""Binary anti-spoofing metrics. Positive class is spoof (label 1)."""
import numpy as np
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)


def compute_metrics(labels, spoof_scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(spoof_scores, dtype=np.float64)
    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and scores must be non-empty and equal length.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation manifest must contain both labels 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinity.")

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1.0 - tpr

# Match the EER calculation used in the B1 training notebook.
    index = int(np.nanargmin(np.abs(fnr - fpr)))

    eer = float((fpr[index] + fnr[index]) / 2.0)
    eer_threshold = float(thresholds[index])

    preds = (scores >= eer_threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "count": int(len(labels)),
        "bonafide_count": int((labels == 0).sum()),
        "spoof_count": int((labels == 1).sum()),
        "eer": eer,
        "eer_percent": 100.0 * eer,
        "eer_threshold_spoof_score": eer_threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "accuracy_at_eer_threshold": float(accuracy_score(labels, preds)),
        "precision_spoof": float(precision_score(labels, preds, zero_division=0)),
        "recall_spoof": float(recall_score(labels, preds, zero_division=0)),
        "f1_spoof": float(f1_score(labels, preds, zero_division=0)),
        "confusion_matrix": {
            "tn_bonafide": int(tn), "fp_bonafide_as_spoof": int(fp),
            "fn_spoof_as_bonafide": int(fn), "tp_spoof": int(tp)
        },
        "roc": {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": [float(x) for x in thresholds]
        }
    }
