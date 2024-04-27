import sys
sys.path.append('../../implmentation')
import numpy as np
import time
from eval_metrics.nmi import nmi
from eval_metrics.ac import ac
from algorithm.nnlsm_BPP import nnlsm_blockpivot


# Symmetric Alternating Nonnegative Least Squares
def SymANLS_pivot(X, U, V, maxiter, lambda_, tol, truelabel = None, print_every=100):

    """ 
    Symmetric alternating nonnegative least squares
    
    Input
    ----------
    X : numpy.array, symmetric matrix (n,n) - matrix need factorization
    U,V : numpy.array or scipy.sparse matrix, shape (n,r) - initialization of the factorization
    maxiter : integer, max # of iteration
    lambda_ : penalty for ||U-V||_2^2
    
    Optional Parameters
    -------------------
    truelabel : np.array (size based on datasets) - truelabel of each element in the dataset

    Output
    -------
    tuple: U, V, e, t, diff, NMI, ACC
    U,V : np.array, shape(n,r) - after SymANLS, factorization of X
    e : np.array, shape corresponding to iteration - loss ||X-UVT||_2^2 / ||X||_2^2
    t : np.array, shape corresponding to iteration - time of each iteration
    diff : np.array, shape corresponding to iteration - difference ||U-V||_2^2
    NMI : np.array, shape corresponding to iteration - measurement for clustering
    ACC : np.array, shape corresponding to iteration - measurement for clustering
    """
    
    # convert V if necessary
    if V.shape[0] < V.shape[1]:
        V = V.T

    n, r = V.shape

    etime = time.time()
    e = []
    t = []
    diff = []
    iter = 0

    # initial error
    e.append(0.5 * np.linalg.norm(X - np.dot(U, V.T), 'fro') ** 2 / np.linalg.norm(X, 'fro') ** 2)
    diff.append(np.linalg.norm(U - V, 'fro') ** 2)
    t.append(0)

    NMI = []
    ACC = []
    if truelabel is not None:
        label = np.argmax(V, axis=1)
        tempacc = nmi(truelabel, label)
        acc = ac(truelabel, label)
        NMI.append(tempacc)
        ACC.append(acc)


    while iter <= maxiter:
        
        # update U
        A = np.concatenate((V, np.sqrt(lambda_) * np.eye(r))) #test_random (25, 5)
        B = np.concatenate((X.T, np.sqrt(lambda_) * V.T)) #test_random (25, 20)
        UT,_ = nnlsm_blockpivot(A, B, 0, U.T) 
        U = UT.T

        # update V
        C = np.concatenate((U, np.sqrt(lambda_) * np.eye(r)))
        D = np.concatenate((X, np.sqrt(lambda_) * U.T))
        VT,_ = nnlsm_blockpivot(C, D, 0, V.T)
        V = VT.T

        # evaluation of the error e at time t
        if e is not None:
            e.append(0.5 * np.linalg.norm(X - np.dot(U, V.T), 'fro') ** 2 / np.linalg.norm(X, 'fro') ** 2)
            diff.append(np.linalg.norm(U - V, 'fro') ** 2)
            t.append(time.time() - etime)

            if truelabel is not None:
                label = np.argmax(V, axis=1)
                tempacc = nmi(truelabel, label)
                acc = ac(truelabel, label)
                NMI.append(tempacc)
                ACC.append(acc)

        iter += 1
        
        if abs(e[-1] - e[-2]) < tol:
            break

        if iter % print_every == 0:
            print(f"iter {iter} done, loss = {e[iter]}, error = {diff[iter]}")
        #if iter % 100 == 0:
        #    print("Iteration:", iter, "Error:", e[iter], "NMI:", NMI[iter])
    # iter -= 1
    # print(iter, len(e), len(t), len(NMI), len(ACC))
    print("last iter", iter, "Error:", e[iter], "NMI:", NMI[-1], "ACC:", ACC[-1])
    return U, V, e, t, diff, NMI[-1], ACC[-1]




