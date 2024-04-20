import sys
sys.path.append('../../implmentation')
import numpy as np
import time
from eval_metrics.nmi import nmi
from eval_metrics.ac import ac
from algorithm.nnlsm_BPP import nnlsm_blockpivot


# Symmetric Alternating Nonnegative Least Squares
def SymANLS_pivot(X, U, V, maxiter, lambda_, truelabel = None):
    
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

        #if iter % 100 == 0:
        #    print("Iteration:", iter, "Error:", e[iter], "NMI:", NMI[iter])
    
    print("last iter", iter, "Error:", e[iter], "NMI:", NMI[iter], "ACC:", ACC[iter])
    return U, V, e, t, diff, NMI




