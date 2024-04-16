# Author: WANG Xiaoyu

import numpy as np
import logging

def kmeans(X, n_clusters, max_iter=100, tol=1e-4):
    """K-means clustering algorithm.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input data.

    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.

    max_iter : int, default=100
        The maximum number of iterations.

    tol : float, default=1e-4
        The relative tolerance with regards to inertia to declare convergence.

    Returns
    -------
    labels : array, shape (n_samples,)
        The label of each sample.

    centers : array, shape (n_clusters, n_features)
        The centers of the clusters.
    """
    n_samples, n_features = X.shape
    centers = X[np.random.choice(n_samples, n_clusters, replace=False)]
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        dist = np.linalg.norm(X[:, None] - centers, axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for i in range(n_clusters):
            centers[i] = X[labels == i].mean(axis=0)
    if _ == max_iter - 1:
        print(f"K-means algorithm did not converge in {max_iter} iterations")
        return None, None
        # raise Exception("K-means algorithm did not converge")
    logging.info(f"K-means algorithm converged in {_} iterations")
    
    return labels, centers


def kmeans_optimal(X, n_clusters, max_iter=200, tol=1e-4, turns=20):
    minimal_distance = np.inf
    
    for _ in range(turns):
        labels, centers = kmeans(X, n_clusters, max_iter, tol)
        if labels is None:
            continue
        distance = np.linalg.norm(X - centers[labels])
        if distance < minimal_distance:
            minimal_distance = distance
            best_labels = labels
            best_centers = centers
    return best_labels, best_centers