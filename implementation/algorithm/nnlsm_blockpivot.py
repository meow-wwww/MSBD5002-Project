import numpy as np

def nnlsm_blockpivot(A, B, isInputProd=0, init=None):
    if isInputProd:
        AtA = A
        AtB = B
    else:
        AtA = np.dot(A.T, A) #test_random (10, 10), init(10, 100)
        AtB = np.dot(A.T, B) #test_random (10, 100)
    
    n, k = AtB.shape
    MAX_BIG_ITER = n * 5
    
    X = np.zeros((n, k))
    if init is None:
        Y = -AtB
        PassiveSet = np.zeros((n, k), dtype=bool)
        numChol = 0
        numEq = 0
    else:
        PassiveSet = init > 0
        X, numChol, numEq = normalEqComb(AtA, AtB, PassiveSet) # X test_random (10, 100)
        #Y = np.dot(AtA, X) - AtB
        Y = AtA @ X - AtB #test_random (10, 100)

    pbar = 3
    P = np.ones(k) * pbar
    Ninf = np.ones(k) * (n + 1)
    
    NonOptSet = (Y < 0) & ~PassiveSet # non-optimal #bool，True: Y<0& PassiveSet False  #test_random (10, 100)
    InfeaSet = (X < 0) & PassiveSet # infeasible ? nonnegative  #test_random (10, 100)
    NotGood = np.sum(NonOptSet) + np.sum(InfeaSet) #test_random 181
    NotOptCols = NotGood > 0
    
    bigIter = 0
    success = 0
    
    while np.any(NotOptCols):
        bigIter += 1
        if MAX_BIG_ITER > 0 and bigIter > MAX_BIG_ITER: #set max_iter for ill-conditioned (numerically unstable) case
            success = 1
            break
        
        Cols1 = NotOptCols & (NotGood < Ninf) #test_random (100,) False
        Cols2 = NotOptCols & (NotGood >= Ninf) & (P >= 1) #test_random (100,) True
        Cols3Ix = np.where(NotOptCols & ~Cols1 & ~Cols2)[0] #[]
        
        if np.any(Cols1):
            P[Cols1] = pbar
            Ninf[Cols1] = NotGood[Cols1]
            PassiveSet[NonOptSet & np.tile(Cols1, (n, 1))] = True
            PassiveSet[InfeaSet & np.tile(Cols1, (n, 1))] = False
        
        if np.any(Cols2):
            P[Cols2] -= 1
            PassiveSet[NonOptSet & np.tile(Cols2, (n, 1))] = True
            PassiveSet[InfeaSet & np.tile(Cols2, (n, 1))] = False
        
        if len(Cols3Ix) > 0:
            for i in range(len(Cols3Ix)):
                Ix = Cols3Ix[i]
                toChange = np.max(np.where(NonOptSet[:, Ix] | InfeaSet[:, Ix]))
                
                if PassiveSet[toChange, Ix]:
                    PassiveSet[toChange, Ix] = False
                else:
                    PassiveSet[toChange, Ix] = True
        
        X[:, NotOptCols], tempChol, tempEq = normalEqComb(AtA, AtB[:, NotOptCols], PassiveSet[:, NotOptCols])
        numChol += tempChol
        numEq += tempEq
        X[np.abs(X) < 1e-12] = 0 #One can uncomment this line for numerical stability
        Y[:, NotOptCols] = AtA @ X[:, NotOptCols] - AtB[:, NotOptCols]
        Y[np.abs(Y) < 1e-12] = 0 #One can uncomment this line for numerical stability
        
        NotOptMask = np.tile(NotOptCols, (n, 1))
        NonOptSet = NotOptMask & (Y < 0) & ~PassiveSet
        InfeaSet = NotOptMask & (X < 0) & PassiveSet
        NotGood = np.sum(NonOptSet) + np.sum(InfeaSet)
        NotOptCols = NotGood > 0
    
    return X, Y, success, numChol, numEq


def normalEqComb(AtA, AtB, PassSet = None):
    if AtB.size == 0:
        Z = np.array([])
        numChol = 0
        numEq = 0
        #return np.array([]), 0, 0
    elif PassSet is None or np.all(PassSet): # 1st:PassSet(10, 100) is None or all True; 2nd (10,1,100)?
        Z = np.linalg.solve(AtA, AtB) #test_random (10, 100)
        numChol = 1
        numEq = AtB.shape[1]
    else:
        #Z = np.zeros(AtB.shape)
        Z = np.zeros_like(AtB)
        m, k1, k2= PassSet.shape

        if k1 == 1:
            if np.any(PassSet):
                Z[PassSet] = np.linalg.solve(AtA[PassSet, PassSet], AtB[PassSet])
                numChol = 1
                numEq = 1
            else:
                numChol = 0
                numEq = 0
        else:
            #sortedPassSet = np.sort(PassSet.T, axis=0)
            #sortedPassSet, sortIx = np.sort(PassSet.T, axis=0).T
            sortIx = np.lexsort(PassSet.T)
            sortedPassSet = PassSet[sortIx, :]

            breaks = np.any(np.diff(sortedPassSet, axis=0), axis=1)
            breakIx = np.concatenate(([0], np.where(breaks)[0] + 1, [k1]))

            if np.any(sortedPassSet[0] == 0):
                startIx = 2
            else:
                startIx = 1

            numChol = 0
            numEq = k1 - breakIx[startIx]

            for k in range(startIx, len(breakIx) - 1):
                #cols = np.argsort(sortedPassSet[breakIx[k]:breakIx[k+1], :].T, axis=0)
                #vars = sortedPassSet[breakIx[k]:breakIx[k+1], cols]
                cols = sortIx[breakIx[k]:breakIx[k + 1]]
                vars = sortedPassSet[breakIx[k]+1, :].T
                #vars = sortedPassSet[breakIx[k]:breakIx[k + 1]].T
                Z[vars, cols] = np.linalg.solve(AtA[vars, vars], AtB[vars, cols])
                numChol += 1
                #numEq += breakIx[k+1] - breakIx[k]

    return Z, numChol, numEq
            
            