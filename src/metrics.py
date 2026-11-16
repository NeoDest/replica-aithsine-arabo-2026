"""Métricas de la sección 5 (pp. 13-17).

- Tarea binaria: separar prompts de ataque y sensibles (positivos) de los benignos.
- Accuracy con IC bootstrap al 95 % sobre 1 000 remuestreos (p. 13).
- FPR = FP/(FP+TN); FNR = FN/(FN+TP) (Tabla 5, p. 14).
- Recall por clase y especificidad en benignos (Tabla 6, p. 15).
- Falsos positivos por subtipo: general vs. hard negative (Tabla 7, p. 15).
- Intervalos de Wilson al 95 % para proporciones del enrutamiento (Tabla 8, p. 16).
"""
import math

import numpy as np


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, bool); y_pred = np.asarray(y_pred, bool)
    tp = int((y_true & y_pred).sum()); tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum()); fn = int((y_true & ~y_pred).sum())
    div = lambda a, b: a / b if b else 0.0
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"accuracy": div(tp + tn, len(y_true)), "precision": p, "recall": r,
            "f1": div(2 * p * r, p + r), "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def bootstrap_accuracy_ci(y_true, y_pred, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true, bool) == np.asarray(y_pred, bool)).astype(float)
    accs = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    return float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (centre - half, centre + half)


def per_category(labels, y_pred, subtypes=None):
    labels = np.asarray(labels); y_pred = np.asarray(y_pred, bool)
    out = {
        "benign_specificity": float((~y_pred[labels == "benign"]).mean()) if (labels == "benign").any() else None,
        "jailbreak_recall": float(y_pred[labels == "jailbreak"].mean()) if (labels == "jailbreak").any() else None,
        "sensitive_recall": float(y_pred[labels == "sensitive"].mean()) if (labels == "sensitive").any() else None,
    }
    if subtypes is not None:
        st = np.asarray(subtypes)
        for name in ("general", "hard_negative"):
            mask = (labels == "benign") & (st == name)
            if mask.any():
                out[f"fp_{name}"] = f"{int(y_pred[mask].sum())}/{int(mask.sum())}"
    return out
