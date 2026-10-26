"""Evaluation metrics for embeddings. Ground-truth labels are used here only, never by the method."""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def knn_accuracy(embd, labels, k=15, n_splits=5, seed=0):
    """Mean kNN classification accuracy (k neighbors, stratified n_splits-fold CV) of labels in the embedding."""
    clf = KNeighborsClassifier(n_neighbors=k)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(np.mean(cross_val_score(clf, embd, labels, cv=cv)))
