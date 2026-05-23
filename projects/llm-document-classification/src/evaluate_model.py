"""
Evaluate Model
==============
Compute classification metrics: accuracy, F1 per class, confusion matrix.
Source data: ArXiv, PubMed, Wikipedia live API corpus.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, X_test, y_test, label_encoder, model_name="model"):
    """Evaluate a single model and save metrics + confusion matrix plot."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(
        y_test, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    # Save JSON metrics
    metrics = {
        "model": model_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class_f1": {
            cls: float(report[cls]["f1-score"])
            for cls in label_encoder.classes_
            if cls in report
        },
    }
    metrics_path = REPORTS_DIR / f"{model_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate_model] Metrics saved to {metrics_path}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    cm_path = REPORTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate_model] Confusion matrix saved to {cm_path}")

    return metrics


def evaluate_all():
    """Evaluate all saved models."""
    X = np.load(PROCESSED_DIR / "X_tfidf.npy")
    y = np.load(PROCESSED_DIR / "y.npy")
    le = joblib.load(PROCESSED_DIR / "label_encoder.pkl")

    # Split consistent with training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = []
    for model_file in MODELS_DIR.glob("*.pkl"):
        model = joblib.load(model_file)
        name = model_file.stem
        metrics = evaluate_model(model, X_test, y_test, le, model_name=name)
        results.append(metrics)

    return results


if __name__ == "__main__":
    evaluate_all()
