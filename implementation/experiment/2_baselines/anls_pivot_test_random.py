import sys
sys.path.append('../../implementation')
import numpy as np
from algorithm.SymANLS_pivot import SymANLS_pivot
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    # Generate some random data for testing
    np.random.seed(42)
    n = 1000
    r = 5
    
    U_true = np.abs(np.random.rand(n, r))
    X = U_true @ U_true.T
    U0 = np.random.rand(n, r)
    V0 = U0.copy()
    maxiter = 1000
    #lambda_ = 0.1
    lambda_values = [0.01, 0.1, 0.5, 1, 5, 10, 39.9]

    truelabel = np.random.randint(0, 3, size = n)
    
    e_values = []
    # Run SymANLS_pivot algorithm
    for lambda_ in lambda_values:
        U, V, e, t, diff, acc = SymANLS_pivot(X, U0, V0, maxiter, lambda_, truelabel)
        e_values.append(e)
    
    
    #U, V, e, t, diff, NMI = SymANLS_pivot(X, U0, V0, maxiter, lambda_, truelabel)

    
    iterations = range(0, 1002)  

    plt.figure()
    for i, e in enumerate(e_values):
        plt.plot(iterations, e, label=f"lambda = {lambda_values[i]}")

    plt.xlabel('Iteration k')
    plt.ylabel('Error E_k')
    plt.legend()
    plt.show()
 
      
    # Print the results
    #("U:", U)
    #print("V:", V)
    #print("Error:", e)
    #print("Time:", t)
    #print("Difference:", diff)
    #print("nmi:", NMI)

