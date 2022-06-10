import numpy as np
def solve1D(K,s,r,Us,Fr):
    N= np.shape(K)[1]
    F=np.zeros([N,1])
    U=np.zeros([N,1])
    U[s] = np.transpose([Us])
    F[r]= np.transpose([Fr])
    Kred=K[np.ix_(r,r)]
    Kvin=K[np.ix_(r,s)]
    U[r]= np.linalg.solve(Kred,F[r]-Kvin.dot(U[s]))
    F[s]=K[s,:].dot(U)
    
    return U,F