import sys
sys.path.append('../../implementation')
from algorithm.SymANLS_pivot import SymANLS_pivot
from data_preprocessing import coil20
import numpy as np
import matplotlib.pyplot as plt

CLUSTER_NUM = 5
r = 5 # equal to the # of clusters
maxiter = 1000
lambda_ = 0.1

mat, truelabel = coil20.read_data(num_of_class=CLUSTER_NUM, random_shuffle=True, 
                              random_choose_class=True)
# matrix processing
mat -= np.mean(mat)
norm = np.linalg.norm(mat, axis=1).reshape(-1, 1)
mat_normalized = mat / norm
X = mat_normalized @ mat_normalized.T

# initialization
U0 = np.random.rand(X.shape[0], r)
V0 = U0.copy()

U, V, e, t, diff, NMI, ACC = SymANLS_pivot(X, U0, V0, maxiter, lambda_, truelabel)


#print("U ：", U.shape)
#print("V ：", V.shape)
#print("error：", e)
#print("time：", t)
#print("difference between U and V：", diff)
#print("NMI：", NMI)

plt.plot(range(len(e)), e)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.show()
