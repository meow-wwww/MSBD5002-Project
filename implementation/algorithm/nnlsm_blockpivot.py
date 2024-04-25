# Author: QIN Shiyue

import numpy as np
"""
This is an unsuccessful program for SymANLS_pivot, there are still something wrong for line 106, 139, etc.
nnlsm_block pivot is an optimization to solve SymANLS since the past version is too slow.
Refrence:
1.Jingu Kim and Haesun Park, Toward Faster Nonnegative Matrix Factorization: A New Algorithm and Comparisons,
 In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (ICDM'08), 353-362, 2008
2."M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450"

Check https://github.com/kimjingu/nonnegfac-python/tree/master for a successful implementation.
"""

def nnlsm_blockpivot(A, B, isInputProd=0, init=None):
    if isInputProd:
        AtA = A
        AtB = B
    else:
        AtA = A.T @ A #test_random (5, 5), init(5, 20)
        AtB = A.T @ B #test_random (5, 20)
    
    n, k = AtB.shape
    MAX_BIG_ITER = n * 5
    
    X = np.zeros((n, k))
    if init is None:
        Y = -AtB
        PassiveSet = np.zeros((n, k), dtype=bool)
        numChol = 0
        numEq = 0
    else:
        PassiveSet = init > 0 #test_random (5, 20)
        X, numChol, numEq = normalEqComb(AtA, AtB, PassiveSet) # test_random X(5, 20) PassiveSet(5, 20)
        #Y = np.dot(AtA, X) - AtB
        Y = AtA @ X - AtB #Y(5, 20)

    pbar = 3
    P = np.full(k, pbar) #P = np.ones(k) * pbar #（20，）
    Ninf = np.full(k, n + 1) #（20，）
    
    NonOptSet = (Y < 0) & ~PassiveSet # non-optimal #bool，True: Y<0& PassiveSet False  # NonOptSet(5, 20)
    InfeaSet = (X < 0) & PassiveSet # infeasible ? nonnegative  # InfeaSet(5, 20)
    NotGood = NonOptSet.sum(axis = 0) + InfeaSet.sum(axis = 0) # NotGood(20,)
    NotOptCols = NotGood > 0 # NotOptCols(20,)
    
    bigIter = 0
    success = 0
    
    while np.any(NotOptCols):
        bigIter += 1
        if MAX_BIG_ITER > 0 and bigIter > MAX_BIG_ITER: #set max_iter for ill-conditioned (numerically unstable) case
            success = 1
            break
        
        Cols1 = NotOptCols & (NotGood < Ninf) # Cols1(20,) 
        Cols2 = NotOptCols & (NotGood >= Ninf) & (P >= 1) # Cols2(20,) 
        Cols3Ix = np.where(NotOptCols & ~Cols1 & ~Cols2)[0] # Cols3Ix[] (0,)
        
        if np.any(Cols1):
            P[Cols1] = pbar
            Ninf[Cols1] = NotGood[Cols1]
            PassiveSet[:, Cols1] = (NonOptSet[:, Cols1] | PassiveSet[:, Cols1]) & ~InfeaSet[:, Cols1]
        
        if np.any(Cols2):
            P[Cols2] -= 1
            PassiveSet[:, Cols2] = (NonOptSet[:, Cols2] | PassiveSet[:, Cols2]) & ~InfeaSet[:, Cols2]
        
        if Cols3Ix.size > 0:
            for i in range(len(Cols3Ix)):
                Ix = Cols3Ix[i]
                toChange = np.max(np.where(NonOptSet[:, Ix] | InfeaSet[:, Ix]))
                
                if PassiveSet[toChange, Ix]:
                    PassiveSet[toChange, Ix] = False
                else:
                    PassiveSet[toChange, Ix] = True
        
        X[:, NotOptCols], tempChol, tempEq = normalEqComb(AtA, AtB[:, NotOptCols], PassiveSet[:, NotOptCols]) #PassiveSet(5,20) AtB(5,20)
        numChol += tempChol
        numEq += tempEq
        X[np.abs(X) < 1e-12] = 0 #One can uncomment this line for numerical stability
        Y[:, NotOptCols] = AtA @ X[:, NotOptCols] - AtB[:, NotOptCols]
        Y[np.abs(Y) < 1e-12] = 0 #One can uncomment this line for numerical stability
        
        NotOptMask = np.tile(NotOptCols, (n, 1))
        NonOptSet = NotOptMask & (Y < 0) & ~PassiveSet
        InfeaSet = NotOptMask & (X < 0) & PassiveSet
        NotGood = NonOptSet.sum(axis = 0) + InfeaSet.sum(axis = 0)
        NotOptCols = NotGood > 0
    
    return X, Y, success, numChol, numEq


def normalEqComb(AtA, AtB, PassSet = None):
    if AtB.size == 0:
        Z = np.array([])
        numChol = 0
        numEq = 0
        #return np.array([]), 0, 0
    elif PassSet is None or np.all(PassSet): # 1st:PassSet(5, 20) ; 2nd ATB(5,8) PassSet(5,8)
        Z = np.linalg.solve(AtA, AtB) #test_random (5, 8)
        numChol = 1
        numEq = AtB.shape[1]
    else:
        #Z = np.zeros(AtB.shape)
        Z = np.zeros_like(AtB)  # Z(5, 8)
        n, k1 = PassSet.shape #5, 8

        if k1 == 1:
            if np.any(PassSet):
                #Z[PassSet] = np.linalg.solve(AtA[PassSet, PassSet], AtB[PassSet]) #PassSet(5, 1) ##not match dimension
                print('PassSet:', PassSet.shape)
                Z[PassSet] = np.linalg.solve(AtA[PassSet][:, PassSet], AtB[PassSet])
                #Z[PassSet] = np.linalg.solve(AtA[PassSet,:][:,PassSet], AtB[PassSet])
                numChol = 1
                numEq = 1
            else:
                numChol = 0
                numEq = 0
        else:
            sorted_indices = np.lexsort(tuple(PassSet.T[:, i] for i in range(PassSet.shape[0]-1, -1, -1)))
            sortIx = sorted_indices #sortIx(8,)
            sortedPassSet = PassSet.T[sortIx] #sortedPassSet(8, 5)
            
            diff_sortedPassSet = np.diff(sortedPassSet, axis=0)
            breaks = np.any(diff_sortedPassSet.T != 0, axis=0) 
            nonzero_indices = np.nonzero(breaks)[0]
            breakIx = np.concatenate(([0], nonzero_indices, [k1])) 

            if np.all(sortedPassSet[0, :] == 0):
                startIx = 2
            else:
                startIx = 1

            numChol = 0
            print('breaks:', breaks)
            print('sortedPassSet:', sortedPassSet)
            print('breakIx:', breakIx)
            print('startIx:', startIx)
            print('k1:', k1)
            numEq = k1 - breakIx[startIx]

            for k in range(startIx, len(breakIx) - 1):
                cols = sortIx[breakIx[k]:breakIx[k + 1]] #test_random (0, 10)   #V(1,3)
                vars = sortedPassSet[breakIx[k], :] #test_random (10,97) #V(1,5)
                Z[vars,:][:,cols] = np.linalg.solve(AtA[vars][:, vars], AtB[vars][:, cols])
                #Z[np.ix_(vars, cols)] = np.linalg.solve(AtA[np.ix_(vars, vars)], AtB[np.ix_(vars, cols)])
                numChol += 1

    return Z, numChol, numEq
