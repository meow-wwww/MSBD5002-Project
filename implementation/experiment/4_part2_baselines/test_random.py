import sys
sys.path.append('D:/hkust/class/5002Data Mining/project/clone_git/MSBD5002-Project')
#sys.path.append('../..')
import numpy as np
from implementation.algorithm.SymANLS import SymANLS
from implementation.algorithm.SymANLS_pivot import SymANLS_pivot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Generate some random data for testing
    np.random.seed(42)
    n = 100
    r = 10
    
    X = np.random.rand(n, n)
    U0 = np.random.rand(n, r)
    V0 = U0.copy()
    maxiter = 100
    lambda_ = 0.1
    #lambda_values = [0.01, 0.1, 0.5, 1, 5, 10, 39.9]

    truelabel = np.random.randint(0, 3, size=100)
    
    #U, V, e, t, diff, NMI = SymANLS(X, U0, V0, maxiter, lambda_, truelabel)
    U, V, e, t, diff, NMI = SymANLS_pivot(X, U0, V0, maxiter, lambda_, truelabel)

    
    # Print the results
    #("U:", U)
    #print("V:", V)
    print("Error:", e)
    print("Time:", t)
    #print("Difference:", diff)
    print("nmi:", NMI)

