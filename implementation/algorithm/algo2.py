# Author: LUO Zixi
import numpy as np

import torch
# added by WANG Xiaoyu
def matrixFac(A, H,
          lamda, 
          num_iter, print_every=100):
    """Algorithm 2. H \in R^{n x k}, no special constraints on H.
    
    Parameters
    ----------
    A: torch.tensor, shape (n_samples, n_samples)
        A symmetric matrix of size n x n. Measures the pairwise similarity between samples.
        
    H: torch.tensor, shape (n_samples, n_clusters)
    
    lambda: float
        
    num_iter : int
    
    Returns
    -------
    H: array, shape (n_samples, n_clusters)
    """
    
    N = A.shape[0]
    
    # assert A is symmetric
    assert torch.allclose(A, A.T, atol=1e-6)
    
    # pre-processing
    D = torch.diag(A.sum(dim=1))
    L = D - A
    M = A - lamda * L
    
    # for test
    M = A
    # seems that this works
    
    """
    function [H,losses] = matrixFac(X, itr, H)
    losses=zeros(itr,1);
    for t=1:itr
        losses(t)=norm(X-H*H','fro')^2;
        step=1/(norm(X-H*H')+2*norm(H'*H));
        H=H-step*(H*H'*H-X*H);
    end
    """
    
    for iter_num in range(num_iter):
        loss = torch.norm(M - H @ H.T, p='fro')**2
        step = 1/(torch.norm(M - H @ H.T, p=2) + 2 * torch.norm(H.T @ H, p=2))
        H = H - step * (H @ H.T @ H - M @ H)

        if (iter_num+1) % print_every == 0:
            print(f"iter {iter_num+1} done, loss = {loss.item()}, step = {step}")
        
    # assert False, "Algorithm 1 did not converge"
    return H

# constraint of non-negative H >= 0^2
def matrixFacNonNegative(M, num_iter, H):
    count = 0
    while count < num_iter:
        grad = 4 * (np.dot(H, np.dot(np.transpose(H), H)) - np.dot(M, H))
        step = 1 / (2*(4 * (np.linalg.norm(np.dot(H, np.transpose(H)) - M)) + 8 * np.linalg.norm(np.dot(np.transpose(H), H))))
        h = H - step * grad
        h[h < 0] = 0
        count += 1
    return H

# constraint of 2-norm of h is 1
def matrixFacUnit(M, num_iter, H):
    count = 0
    while count < num_iter:
        grad = 4 * (np.dot(H, np.dot(np.transpose(H), H)) - np.dot(M, H))
        step = 1 / (2 * (4 * (np.linalg.norm(np.dot(H, np.transpose(H)) - M)) + 8 * np.linalg.norm(np.dot(np.transpose(H), H))))
        h = (H - step * grad) / np.linalg.norm(H - step * grad)
        count += 1
    return h
