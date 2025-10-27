
# src/eval_utils.py
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)

def compute_metrics(y_true, y_pred, labels=None, label_names=None):
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=label_names, zero_division=0, output_dict=True
    )
    return {
        "accuracy": acc,
        "macro": {"precision": macro[0], "recall": macro[1], "f1": macro[2]},
        "micro": {"precision": micro[0], "recall": micro[1], "f1": micro[2]},
        "weighted": {"precision": weighted[0], "recall": weighted[1], "f1": weighted[2]},
        "per_class": {
            (label_names[i] if label_names else str(l)): {
                "precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(support[i])
            } for i, l in enumerate(labels or sorted(set(y_true)))
        },
        "classification_report": report
    }

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def plot_confusion(y_true, y_pred, label_names=None, normalize=False, title=None, save_path=None):
    cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    if title:
        ax.set_title(title)
    ticks = range(len(cm))
    if label_names is None:
        label_names = [str(i) for i in ticks]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_yticklabels(label_names)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha='center', va='center', fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    return fig, ax
