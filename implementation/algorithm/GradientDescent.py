# Author: WANG Xiaoyu
import torch
import torch.optim as optim

def GradientDescent(A, H, lamda, 
                     loss_fn,
                     lr,
                     max_iter=100, tol=1e-4, print_every=100):
    '''
    A: a symmetric matrix of size n x n
    H: a matrix of size n x k
    lambda_: a scalar
    '''
    # optimizer = optim.Adam([H], lr=lr)
    optimizer = optim.SGD([H], lr=lr)
    
    assert H.requires_grad, "H is not set to require grad"
    
    optimizer.zero_grad()
        
    loss_last = 0
    for i in range(max_iter):
        loss = loss_fn(A, H, lamda)
        loss.backward()
        with torch.no_grad():
            # H /= torch.norm(H, dim=1).reshape(-1, 1)
            optimizer.step()
            optimizer.zero_grad()
        
        if i % print_every == 0:
            print(f'iteration {i}, loss {loss.item()}')
        if abs(loss - loss_last) < tol:
            break
        else:
            loss_last = loss
    return H, loss