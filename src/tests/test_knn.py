from sklearn.datasets import make_classification
from src.etl.knn import perform_k_fold_validation


def test_perform_k_fold_validation():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )

    n_splits = 10
    n_neighbors = 3
    metric = "euclidean"

    results = perform_k_fold_validation(X, y, n_splits, n_neighbors, metric)

    assert isinstance(results, dict)
    assert "accuracy" in results
    assert "auc" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results

    assert len(results["accuracy"]) == n_splits
    assert len(results["auc"]) == n_splits
    assert len(results["precision"]) == n_splits
    assert len(results["recall"]) == n_splits
    assert len(results["f1"]) == n_splits
