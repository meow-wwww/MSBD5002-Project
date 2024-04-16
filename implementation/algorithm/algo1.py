# Author: zrb

import numpy as np
import time 
def matrixFac_col(X, itr, H, lmbda, tol):
    """
    -------------
    Parameters
    -------------
    X: Symmetric input matrix of size (n, n)
    itr: Number of iterations for the factorization
    H: Initial matrix H of shape (n, k)
    lmbda: Regularization parameter
    tol: Tolerance to determine convergence
    -------------
    Output
    -------------
    H: Learned matrix H of shape (n, k)
    P: Learned matrix P of shape (n, k)
    losses: Loss values at each iteration
    losses_pure: Pure reconstruction loss values at each iteration
    times: Time elapsed at each iteration
    H_list: List of H matrices at each iteration
    """
    losses = np.zeros(itr)
    losses_pure = np.zeros(itr)
    H_list = [None] * itr
    etime = time.time()
    times = np.zeros(itr)
    M = X.copy()
    P = H.copy()
    _, k = H.shape
    func_def = lambda a, b: np.dot(a, b.T)
    times[0] = 0
    losses[0] = 0.5 * (np.linalg.norm(X - np.dot(H, P.T), 'fro')**2 + lmbda * np.linalg.norm(H - P, 'fro')**2) / np.linalg.norm(X, 'fro')**2
    losses_pure[0] = 0.5 * (np.linalg.norm(X - np.dot(H, P.T), 'fro')**2) / np.linalg.norm(X, 'fro')**2
    for t in range(itr):
        for i in range(k):
            C = func_def(H, P)
            M_bar = M - (C - np.outer(H[:, i], P[:, i]))
            H[:, i] = np.dot((M_bar + lmbda * np.eye(M_bar.shape[0])), P[:, i]) / (np.linalg.norm(P[:, i])**2 + lmbda)
            P[:, i] = np.dot((M_bar + lmbda * np.eye(M_bar.shape[0])), H[:, i]) / (np.linalg.norm(H[:, i])**2 + lmbda)
        losses[t+1] = 0.5 * (np.linalg.norm(X - np.dot(H, P.T), 'fro')**2 + lmbda * np.linalg.norm(H - P, 'fro')**2) / np.linalg.norm(X, 'fro')**2
        losses_pure[t+1] = 0.5 * (np.linalg.norm(X - np.dot(H, P.T), 'fro')**2) / np.linalg.norm(X, 'fro')**2
        times[t+1] = time.time() - etime
        H_list[t] = H.copy()
        if t > 0:
            if losses_pure[t-1] - losses_pure[t] < tol:
                break

    return H, P, losses, losses_pure, times, H_list
