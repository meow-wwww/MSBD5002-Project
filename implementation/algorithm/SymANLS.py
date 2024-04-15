# Author: QIN Shiyue

import numpy as np
from scipy.optimize import nnls #nnls only accept the b as a vector
from sklearn.metrics import normalized_mutual_info_score
import time
import cvxpy as cp

def ClusteringMeasure(true_labels, pred_labels):
    nmi_ = normalized_mutual_info_score(true_labels, pred_labels)
    return [nmi_]

# Symmetric Alternating Nonnegative Least Squares
def SymANLS(X, U, V, maxiter, lambda_, truelabel = None):
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

    nmi = []
    if truelabel is not None:
        label = np.argmax(V, axis=1)
        tempacc = ClusteringMeasure(truelabel, label)
        nmi.append(tempacc[0])


    while iter <= maxiter:
        
        # update U
        A = np.concatenate((V, np.sqrt(lambda_) * np.eye(r)))
        B = np.concatenate((X.T, np.sqrt(lambda_) * V.T))
        UT_optimal = cp.Variable((U.T).shape, nonneg=True)
        object1 = cp.Minimize(cp.sum_squares(A @ UT_optimal - B))
        constraint1 = []
        prob1 = cp.Problem(object1, constraint1)
        prob1.solve() # solver = SCS
        UT = UT_optimal.value
        U = UT.T

        # update V
        C = np.concatenate((U, np.sqrt(lambda_) * np.eye(r)))
        D = np.concatenate((X, np.sqrt(lambda_) * U.T))
        VT_optimal = cp.Variable((V.T).shape, nonneg=True)
        object2 = cp.Minimize(cp.sum_squares(C @ VT_optimal - D))
        constraint2 = []
        prob2 = cp.Problem(object2, constraint2)
        prob2.solve() # solver = SCS
        VT = VT_optimal.value
        V = VT.T

        # evaluation of the error e at time t
        if e is not None:
            e.append(0.5 * np.linalg.norm(X - np.dot(U, V.T), 'fro') ** 2 / np.linalg.norm(X, 'fro') ** 2)
            diff.append(np.linalg.norm(U - V, 'fro') ** 2)
            t.append(time.time() - etime)

            if truelabel is not None:
                #_, label = pairwise_distances_argmin_min(V, metric = 'euclidean')
                label = np.argmax(V, axis=1)
                tempacc = ClusteringMeasure(truelabel, label)
                nmi.append(tempacc[0])

        iter += 1

    return U, V, e, t, diff, nmi

#may need check error Ek in the article
#checking optimality(FOR PHASE TRASITION ONLY)
#     E_U = max(U - (U*V'-X)*V + lambda*(U-V), 0) - U;
#     E_V = max(V - (U*V'-X)'*U + lambda*(V-U), 0) - V;
#     if norm([E_U;E_V],'fro')< 1e-8
#         disp('get optimality');
#         break;
#     end



