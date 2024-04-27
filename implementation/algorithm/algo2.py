# Author: LUO Zixi
import numpy as np
import torch.nn.functional as F
import torch

# added by WANG Xiaoyu
def matrixFac(A, H,
              tol,
              lamda=1,
              max_iter=10000000, print_every=100):
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
    
    error_last = -1
    
    for iter_num in range(max_iter):
        loss = torch.norm(M - H @ H.T, p='fro')**2
        step = 1/(torch.linalg.norm(M - H @ H.T, ord=2) + 2 * torch.linalg.norm(H.T @ H, ord=2))
        H = H - step * (H @ H.T @ H - M @ H)
        
        error = torch.norm(M - H @ H.T, p='fro')**2 / torch.norm(M, p='fro')**2
        
        if (iter_num+1) % print_every == 0:
            print(f"iter {iter_num+1} done, loss = {loss.item()}, error = {error}")
            
        if abs(error - error_last) < tol:
            break
        else:
            error_last = error
        
    # assert False, "Algorithm 1 did not converge"
    return H

# constraint of non-negative H >= 0^2
def matrixFacNonNegative(A, H, lamda, K, print_every):
    '''D = torch.diag(A.sum(dim=1))
    L = D - A
    M = A - lamda * L'''
    M = A
    
    for iter in range(K):
        loss = torch.norm(M - H @ H.T, p='fro')**2
        step = 1/(torch.linalg.norm(M - H @ H.T, ord=2) + 2 * torch.linalg.norm(H.T @ H, ord=2))
        H = H - step * (H @ H.T @ H - M @ H)
        H[H < 0] = 0

        if (iter+1) % print_every == 0:
            print(f"iter {iter+1} done, loss = {loss.item()}, step = {step}")

    return H

# constraint of 2-norm of h is 1
def matrixFacUnit(A, H, lamda, K, print_every):
    #D = torch.diag(A.sum(dim=1))
    #L = D - A
    #M = A - lamda * L
    M = A

    for iter in range(K):
        loss = torch.linalg.norm(M - H @ H.T, ord='fro')**2
        step = 1/(torch.linalg.norm(M - H @ H.T, ord=2) + 2 * torch.linalg.norm(H.T @ H, ord=2))
        H = H - step * (H @ H.T @ H - M @ H)
        H = F.normalize(H, p=2, dim=1)

        if (iter+1) % print_every == 0:
            print(f"iter {iter+1} done, loss = {loss.item()}, step = {step}")
    
    return H
