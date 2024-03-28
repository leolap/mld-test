from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from tabulate import tabulate


def perform_k_fold_validation(X, y, n_splits, n_neighbors, metric="euclidean"):
    kf = StratifiedKFold(n_splits=n_splits)
    results = {"accuracy": [], "auc": [], "precision": [], "recall": [], "f1": []}

    y_binarized = label_binarize(y, classes=np.unique(y))

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_binarized[train_index], y_binarized[test_index]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)
        y_pred_proba_list = knn.predict_proba(X_test)

        if isinstance(y_pred_proba_list, list):
            y_pred_proba = np.column_stack(y_pred_proba_list)
        else:
            y_pred_proba = y_pred_proba_list

        results["accuracy"].append(
            accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        )

        precision_recall_auc_scores = []
        num_classes = y_test.shape[1]
        for i in range(num_classes):
            precision, recall, _ = precision_recall_curve(
                y_test[:, i], y_pred_proba[:, i]
            )
            pr_auc = auc(recall, precision)
            precision_recall_auc_scores.append(pr_auc)
        avg_precision_recall_auc = np.mean(precision_recall_auc_scores)
        results["auc"].append(avg_precision_recall_auc)

        results["precision"].append(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        results["recall"].append(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )
        results["f1"].append(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return results


def extract_metrics(folds: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Fold": range(1, folds + 1),
        "Accuracy": metrics["accuracy"],
        "AUC": metrics["auc"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1-Score": metrics["f1"],
    }


def write_pdf_comparison(
    folds: int, dict_cosine: Dict[str, Any], dict_euclidean: Dict[str, Any]
):
    metrics_cosine = extract_metrics(folds, dict_cosine)

    metrics_euclidean = extract_metrics(folds, dict_euclidean)

    df_cosine = pd.DataFrame(metrics_cosine)
    df_euclidean = pd.DataFrame(metrics_euclidean)

    avg_metrics_cosine = df_cosine.drop(columns=["Fold"]).mean().to_dict()
    avg_metrics_euclidean = df_euclidean.drop(columns=["Fold"]).mean().to_dict()

    table_cosine = tabulate(df_cosine, headers="keys", tablefmt="grid")
    table_euclidean = tabulate(df_euclidean, headers="keys", tablefmt="grid")

    with open("comparison_tables.txt", "w") as f:
        f.write("Cosine Distance Metrics:\n")
        f.write(table_cosine)
        f.write("\n\nEuclidean Distance Metrics:\n")
        f.write(table_euclidean)

    with PdfPages("comparison_tables.pdf") as pdf:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cosine.set_index("Fold"), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Cosine Distance Metrics")
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            df_euclidean.set_index("Fold"), annot=True, fmt=".2f", cmap="coolwarm"
        )
        plt.title("Euclidean Distance Metrics")
        pdf.savefig()
        plt.close()

        avg_metrics = pd.DataFrame(
            [avg_metrics_cosine, avg_metrics_euclidean], index=["Cosine", "Euclidean"]
        )
        plt.figure(figsize=(8, 3))
        sns.heatmap(avg_metrics, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Average Metrics Comparison")
        pdf.savefig()
        plt.close()


def knn_predict_proba(X, y, metric="euclidean", n=5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    knn_classifier = KNeighborsClassifier(n_neighbors=n, metric=metric)
    knn_classifier.fit(X_train, y_train)

    y_pred_proba = knn_classifier.predict_proba(X_test)

    return y_test, y_pred_proba


def multi_class_roc_curve(y_true, y_pred_proba):
    n_classes = y_pred_proba.shape[1]
    tprs = []
    roc_auc = []

    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

    all_fpr = np.unique(
        np.concatenate([np.linspace(0, 1, 100) for _ in range(n_classes)])
    )

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc.append(auc(fpr, tpr))
        tprs.append(np.interp(all_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(all_fpr, mean_tpr)

    return all_fpr, mean_tpr, mean_auc


def plot_multiclass_roc_curve(y_true_1, y_pred_proba_1, y_true_2, y_pred_proba_2):

    all_fpr_1, mean_tpr_1, mean_auc_1 = multi_class_roc_curve(y_true_1, y_pred_proba_1)
    all_fpr_2, mean_tpr_2, mean_auc_2 = multi_class_roc_curve(y_true_2, y_pred_proba_2)

    plt.figure(figsize=(8, 6))
    plt.plot(
        all_fpr_1,
        mean_tpr_1,
        label=f"Metric 1 mean ROC curve (area = {mean_auc_1:.2f})",
    )
    plt.plot(
        all_fpr_2,
        mean_tpr_2,
        label=f"Metric 2 mean ROC curve (area = {mean_auc_2:.2f})",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Multi-Class")
    plt.legend(loc="lower right")
    plt.show()
