# Author: zrb

import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time 
def matrixFac_col(X, H, lmbda, tol, itr, print_every=100):
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
    M = X.clone()
    P = H.clone()
    _, k = H.shape
    times[0] = 0
    losses[0] = 0.5 * (torch.linalg.norm(X - H @ P.T, 'fro')**2 + lmbda * torch.linalg.norm(H - P, 'fro')**2) / torch.linalg.norm(X, 'fro')**2
    losses_pure[0] = 0.5 * (torch.linalg.norm(X - H @ P.T, 'fro')**2) / torch.linalg.norm(X, 'fro')**2
    for t in range(itr):
        for i in range(k):
            C = H @ P.T
            M_bar = M - (C - H[:, i] @ P[:, i].T)
            H[:, i] = torch.from_numpy(torch.dot((M_bar + lmbda * torch.eye(M_bar.shape[0])), P[:, i]) / (torch.linalg.norm(P[:, i])**2 + lmbda)).to(device)
            P[:, i] = torch.from_numpy(torch.dot((M_bar + lmbda * torch.eye(M_bar.shape[0])), H[:, i]) / (torch.linalg.norm(H[:, i])**2 + lmbda)).to(device)
        losses[t+1] = 0.5 * (torch.linalg.norm(X - torch.dot(H, P.T), 'fro')**2 + lmbda * torch.linalg.norm(H - P, 'fro')**2) / torch.linalg.norm(X, 'fro')**2
        losses_pure[t+1] = 0.5 * (torch.linalg.norm(X - torch.dot(H, P.T), 'fro')**2) / torch.linalg.norm(X, 'fro')**2
        times[t+1] = time.time() - etime
        H_list[t] = H.clone()
        if t > 0:
            if abs(losses_pure[t-1] - losses_pure[t]) < tol:
                break
        if (t+1) % print_every == 0:
            print(f"Iteration {t+1} done, update = {losses_pure[t-1] - losses_pure[t]}")

    return H, P, losses, losses_pure, times, H_list


# Author: WANG Xiaoyu

def algo1(A, H,
          lamda, 
          max_iter, tol, print_every=100):
    """Algorithm 1.
    
    Parameters
    ----------
    A: torch.tensor, shape (n_samples, n_samples)
        A symmetric matrix of size n x n.
        
    H: torch.tensor, shape (n_samples, n_clusters)
        
    k: int
        The number of clusters to form as well as the number of centroids to generate.
    
    lamda : float
        The regularization parameter.
        
    max_iter : int, default=100
    
    tol : float, default=1e-4
        The relative tolerance with regards to inertia to declare convergence.
    
    Returns
    -------
    H: array, shape (n_samples, n_clusters)
    
    loss: final loss, float
    """
    N = A.shape[0]
    
    # assert A is symmetric
    assert torch.allclose(A, A.T, atol=1e-6)
    
    # D = torch.diag(A.sum(dim=1))
    # L = D - A
    # M = A - lamda * L
    M = A
    
    # P is a copy of H
    P = H.clone()
    
    assert (H-P).sum() < 1e-6
    
    loss_old = 100
    
    etime = time.time()
    t = [0]
    e = []
    
    e.append(0.5 * torch.norm(M - H @ P.T)**2 / torch.norm(M)**2)
    
    for iter_num in range(max_iter):
        for i in range(H.shape[1]):
            # compute \Sum_{j!=i}h_jp_j^T
            sum_hp = H @ P.T - H[:, i:i+1] @ P[:, i:i+1].T
            M_bar = M - sum_hp
            H[:, i:i+1] = ((M_bar + lamda * torch.eye(N)) @ P[:, i:i+1]) / (P[:, i:i+1].T @ P[:, i:i+1] + lamda)
            P[:, i:i+1] = ((M_bar + lamda * torch.eye(N)) @ H[:, i:i+1]) / (H[:, i:i+1].T @ H[:, i:i+1] + lamda)
            # assert (H-P).sum() < 1e-6
            
        t.append(time.time() - etime)
        
        loss = 0.5 * torch.norm(M - H @ P.T)**2 / torch.norm(M)**2
        
        e.append(loss)
        
        if abs(loss_old - loss) < tol:
            return H, P, t, e
        elif not abs(loss_old - loss) == abs(loss_old - loss):
            print('NaN detected')
            return None, None, t, e
        else:
            if (iter_num+1) % print_every == 0:
                print(f"iter {iter_num+1} done, update = {abs(loss_old-loss)}")
            loss_old = loss
        
    # assert False, "Algorithm 1 did not converge"
    return H, P, t, e




def algo1_np(A, H,
          lamda, 
          max_iter, tol, print_every=100):
    """Algorithm 1.
    
    Parameters
    ----------
    A: torch.tensor, shape (n_samples, n_samples)
        A symmetric matrix of size n x n.
        
    H: torch.tensor, shape (n_samples, n_clusters)
        
    k: int
        The number of clusters to form as well as the number of centroids to generate.
    
    lamda : float
        The regularization parameter.
        
    max_iter : int, default=100
    
    tol : float, default=1e-4
        The relative tolerance with regards to inertia to declare convergence.
    
    Returns
    -------
    H: array, shape (n_samples, n_clusters)
    
    loss: final loss, float
    """
    N = A.shape[0]
    
    # assert A is symmetric
    # assert torch.allclose(A, A.T, atol=1e-6)
    
    # D = torch.diag(A.sum(dim=1))
    # L = D - A
    # M = A - lamda * L
    M = A
    
    # P is a copy of H
    P = H.copy()
    
    assert (H-P).sum() < 1e-6
    
    loss_old = 100
    
    etime = time.time()
    t = [0]
    e = []
    
    e.append(0.5 * np.linalg.norm(M - H @ P.T)**2 / np.linalg.norm(M)**2)
    
    for iter_num in range(max_iter):
        for i in range(H.shape[1]):
            # compute \Sum_{j!=i}h_jp_j^T
            sum_hp = H @ P.T - H[:, i:i+1] @ P[:, i:i+1].T
            M_bar = M - sum_hp
            H[:, i:i+1] = ((M_bar + lamda * np.eye(N)) @ P[:, i:i+1]) / (P[:, i:i+1].T @ P[:, i:i+1] + lamda)
            P[:, i:i+1] = ((M_bar + lamda * np.eye(N)) @ H[:, i:i+1]) / (H[:, i:i+1].T @ H[:, i:i+1] + lamda)
            # assert (H-P).sum() < 1e-6
            
        t.append(time.time() - etime)
        
        loss = 0.5 * np.linalg.norm(M - H @ P.T)**2 / np.linalg.norm(M)**2
        
        e.append(loss)
        
        if abs(loss_old - loss) < tol:
            return H, P, t, e
        elif not abs(loss_old - loss) == abs(loss_old - loss):
            print('NaN detected')
            return None, None, t, e
        else:
            if (iter_num+1) % print_every == 0:
                print(f"iter {iter_num+1} done, update = {abs(loss_old-loss)}")
            loss_old = loss
        
    # assert False, "Algorithm 1 did not converge"
    return H, P, t, e



def algo1_np_optimized(A, H,
          lamda, 
          max_iter, tol, print_every=100):
    """Algorithm 1.
    
    Parameters
    ----------
    A: torch.tensor, shape (n_samples, n_samples)
        A symmetric matrix of size n x n.
        
    H: torch.tensor, shape (n_samples, n_clusters)
        
    k: int
        The number of clusters to form as well as the number of centroids to generate.
    
    lamda : float
        The regularization parameter.
        
    max_iter : int, default=100
    
    tol : float, default=1e-4
        The relative tolerance with regards to inertia to declare convergence.
    
    Returns
    -------
    H: array, shape (n_samples, n_clusters)
    
    loss: final loss, float
    """
    N = A.shape[0]
    
    M = A
    
    # P is a copy of H
    P = H.copy()
    
    assert (H-P).sum() < 1e-6
    
    loss_old = 100
    
    # HPT = H @ P.T
    M_NORM_SQUARE = np.linalg.norm(M) ** 2
    
    print('M_NORM_SQUARE', M_NORM_SQUARE)
    
    etime = time.time()
    t = [0]
    e = []
    
    e.append(0.5 * np.linalg.norm(M - H @ P.T)**2 / M_NORM_SQUARE)
    
    for iter_num in range(max_iter):
        for i in range(H.shape[1]):
            # compute \Sum_{j!=i}h_jp_j^T
            sum_hp = H @ P.T - H[:, i:i+1] @ P[:, i:i+1].T
            M_bar = M - sum_hp
            H[:, i:i+1] = ((M_bar + lamda * np.eye(N)) @ P[:, i:i+1]) / (P[:, i:i+1].T @ P[:, i:i+1] + lamda)
            P[:, i:i+1] = ((M_bar + lamda * np.eye(N)) @ H[:, i:i+1]) / (H[:, i:i+1].T @ H[:, i:i+1] + lamda)
            # assert (H-P).sum() < 1e-6
            
        t.append(time.time() - etime)
        
        # HPT = H @ P.T
        
        loss = 0.5 * np.linalg.norm(M - H @ P.T)**2 / M_NORM_SQUARE
        
        e.append(loss)
        
        if abs(loss_old - loss) < tol:
            return H, P, t, e
        elif not abs(loss_old - loss) == abs(loss_old - loss):
            print('NaN detected')
            return None, None, t, e
        else:
            if (iter_num+1) % print_every == 0:
                print(f"iter {iter_num+1} done, update = {abs(loss_old-loss)}")
            loss_old = loss
        
    # assert False, "Algorithm 1 did not converge"
    return H, P, t, e