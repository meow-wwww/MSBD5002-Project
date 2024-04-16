# Author: TANG Zhongyi

import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix
from scipy import sparse
from scipy.sparse.linalg import eigsh, lobpcg
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from sklearn.utils.validation import check_random_state, check_array, check_symmetric


def spectral_clustering(X, labels, n_clusters, affinity='rbf', gamma=1., n_neighbors=10):

    """
    Perform spectral clustering from features or affinity matrix.

    Parameters:
    - X : array-like of shape (n_samples, n_features)
        Input data.
    - labels : array-like of shape (n_samples,)
        Labels for each sample, used for computing class weights.
    - n_clusters : int
        The dimension of the projection subspace.
    - affinity : str, default='rbf'
        How to construct the affinity matrix.
         'rbf' : use the Gaussian kernel.
         'nearest_neighbors' : construct the affinity matrix by computing a graph of nearest neighbors.
    - gamma : float, default=1.0
        Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels. Ignored for 'nearest_neighbors'.
    - n_neighbors : int, default=10
        Number of neighbors to use when constructing the affinity matrix using the 'nearest_neighbors' method.

    Returns:
    - predicted_labels : ndarray of shape (n_samples,)
        Labels of each point
    """

    # Compute class weights
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    class_weights_dict = dict(zip(unique_labels, class_weights))

    def create_affinity_matrix(X, labels, class_weights_dict, affinity, gamma=1., n_neighbors=10, include_self=True):

        """
        Create an affinity matrix based on the selected method (RBF or nearest neighbors).
        """

        if affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=include_self)
            affinity_matrix = 0.5 * (connectivity + connectivity.T)
        else:
            affinity_matrix = pairwise_kernels(X, metric=affinity, filter_params=True, gamma=gamma)

        affinity_matrix = lil_matrix(affinity_matrix)

        # Apply class weights to the non-diagonal elements of the matrix
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i != j:
                    weight = class_weights_dict.get(label_i, 1) * class_weights_dict.get(label_j, 1)
                    affinity_matrix[i, j] *= weight

        return affinity_matrix.tocsr()

    def set_diag(laplacian, value, norm_laplacian):
        if not sparse.isspmatrix(laplacian):
            laplacian.flat[::laplacian.shape[0] + 1] = value
        else:
            if norm_laplacian:
                laplacian = laplacian.tocoo()
                diag_idx = (laplacian.row == laplacian.col)
                laplacian.data[diag_idx] = value
                laplacian = laplacian.tocsr()
        return laplacian

    def spectral_embedding(adjacency, n_components, eigen_solver=None, random_state=None,
                           eigen_tol=0.0, norm_laplacian=True):

        """
        Compute the spectral embedding and perform k-means clustering.
        """

        random_state = check_random_state(random_state)
        adjacency = check_symmetric(adjacency)

        if eigen_solver is None:
            eigen_solver = 'arpack' if adjacency.shape[0] > 5 * n_components else 'lobpcg'

        n_nodes = adjacency.shape[0]

        laplacian, dd = csgraph_laplacian(adjacency, normed=norm_laplacian, return_diag=True)
        laplacian *= -1

        if eigen_solver == 'arpack':
            v0 = random_state.uniform(-1, 1, laplacian.shape[0])
            try:
                lambdas, diffusion_map = eigsh(laplacian, k=n_components, sigma=1.0, which='LM',
                                               tol=eigen_tol, v0=v0)
                embedding = diffusion_map.T * dd

                embedding = embedding[:n_components].T
            except RuntimeError as e:
                # Handling the case when eigsh fails
                print(f"ARPACK solver failed: {e}, switching to lobpcg solver.")
                eigen_solver = 'lobpcg'

        if eigen_solver == 'lobpcg':
            laplacian = laplacian * -1
            laplacian = set_diag(laplacian, 1, norm_laplacian)
            laplacian = check_array(laplacian, dtype=np.float64, accept_sparse=True)
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()
            lambdas, diffusion_map = lobpcg(laplacian, X, tol=eigen_tol, largest=False, maxiter=2000)
            embedding = diffusion_map.T * dd

            embedding = embedding[:n_components].T

        # Make the signs of the eigenvectors deterministic
        max_abs_rows = np.argmax(np.abs(embedding), axis=1)
        signs = np.sign(embedding[np.arange(n_nodes), max_abs_rows])
        embedding *= signs[:, np.newaxis]

        return embedding

    # Function to perform the spectral clustering algorithm
    def spectral_clustering_algorithm(affinity, n_clusters):

        """
        Compute the spectral embedding and perform k-means clustering.
        """

        n_components = n_clusters
        maps = spectral_embedding(affinity, n_components=n_components)
        kmeans = KMeans(n_clusters=n_clusters).fit(maps)
        labels = kmeans.labels_
        return labels

    # Generate the affinity matrix from the input data and labels
    affinity_matrix = create_affinity_matrix(X, labels, class_weights_dict, affinity, gamma, n_neighbors)
    predicted_labels = spectral_clustering_algorithm(affinity_matrix, n_clusters)
    return predicted_labels
