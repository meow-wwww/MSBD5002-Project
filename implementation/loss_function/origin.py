# Author: WANG Xiaoyu

import torch

def reconstruction_loss_without_regularization(A, H, lambda_=None):
    '''
    A: torch.tensor, a symmetric matrix of size n x n
    H: torch.tensor, a matrix of size n x k
    '''
    assert A.T.equal(A), "A is not symmetric"
    assert lambda_ is None, "lambda_ should be None"
    
    return torch.norm(A - H @ H.T) ** 2

def reconstruction_loss_with_regularization(A, H, lambda_):
    '''
    A: torch.tensor, a symmetric matrix of size n x n
    H: torch.tensor, a matrix of size n x k
    lambda_: a scalar
    '''
    assert A.T.equal(A), "A is not symmetric"
    
    loss = torch.norm(A - H @ H.T) ** 2
    
    # Vectorized computation of regularization loss
    diff = H[:, None, :] - H[None, :, :]
    regularization_loss = (A.unsqueeze(-1) * diff**2).sum()
    regularization_loss *= lambda_
    
    return loss + regularization_loss