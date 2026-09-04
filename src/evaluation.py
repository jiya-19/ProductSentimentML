"""
evaluation.py

Evaluation and visualization helpers: classification metrics, confusion
matrices, model-comparison charts and linear-model feature importance.

Macro-averaged precision/recall/F1 is used as the headline multiclass
metric throughout this project (rather than accuracy or a
micro/weighted average) because the sentiment classes are imbalanced
(far more Positive reviews than Neutral or Negative). Accuracy alone
would reward a model that mostly predicts "Positive"; macro-averaging
weights each class equally, so poor performance on the minority
Negative/Neutral classes is not hidden by strong performance on the
majority Positive class.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABELS = ["Negative", "Neutral", "Positive"]


def compute_metrics(y_true, y_pred, average: str = "macro") -> dict:
    """Compute accuracy, precision, recall and F1 (macro-averaged by
    default) for a set of predictions."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "Recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "F1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def compute_per_class_metrics(y_true, y_pred, labels=LABELS) -> pd.DataFrame:
    """Per-class precision/recall/F1 — used to identify which sentiment
    class is hardest for a given model (Research Question 5)."""
    prec = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return pd.DataFrame({"Precision": prec, "Recall": rec, "F1": f1}, index=labels)


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str = None):
    """Plot a labeled confusion matrix for one model's predictions."""
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        cbar=True,
    )
    ax.set_xlabel("Predicted Sentiment")
    ax.set_ylabel("Actual Sentiment")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Grouped bar chart comparing Accuracy/Precision/Recall/F1 across
    models. `results_df` must be indexed by model name with those four
    columns."""
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    fig, ax = plt.subplots(figsize=(9, 5))
    results_df[metrics].plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.set_title("Model Comparison on Held-Out Test Set")
    ax.set_ylim(0, 1)
    ax.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def get_top_features_linear(pipeline, top_n: int = 20) -> dict:
    """Extract the top-N most influential TF-IDF features per class for a
    linear model (LogisticRegression or LinearSVC) inside a fitted
    Pipeline named `tfidf` + `clf`.

    Returns {class_label: DataFrame(feature, weight)}.
    Note: these are features the *model* learned to associate with a
    class based on the training data, not universal linguistic truths
    about sentiment.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(vectorizer.get_feature_names_out())

    coefs = clf.coef_  # shape: (n_classes, n_features) for multiclass
    classes = clf.classes_

    top_features = {}
    for i, cls in enumerate(classes):
        row = coefs[i]
        top_idx = np.argsort(row)[-top_n:][::-1]
        top_features[cls] = pd.DataFrame(
            {"feature": feature_names[top_idx], "weight": row[top_idx]}
        )
    return top_features
