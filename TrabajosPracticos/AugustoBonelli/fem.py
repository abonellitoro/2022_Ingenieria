import numpy as np

def solve(k, s,r, us, fr):
	N = k.shape[1]
	f = np.zeros([N, 1])
	u = np.zeros([N, 1])

	u[s] = np.transpose([us])
	f[r] = np.transpose([fr])

	k_redux = k[np.ix_(r,r)]
	k_vinculos = k[np.ix_(r,s)]
	u[r] = np.linalg.solve(k_redux, f[r]-k_vinculos.dot(u[s]))
	f[s] = k[s,:].dot(u)
	
	return u, f

def k_elemental(MN, MC, Ee, Ae, element):
    Lx = MN[MC[element, 1], 0] -MN[MC[element, 0], 0]
    Lx = MN[MC[element, 1], 1] -MN[MC[element, 0], 1]
    L = np.sqrt(Lx**2+Ly**2)
    phi = np.arctan2(Ly, Lx)
    
    cos = np.cos(phi)
    sen = np.sin(phi)
    
    
    k_elemental = (Ee*Ae/L)*np.array([[cos**2,cos*sen,-cos**2,-cos*sen],
                             [cos*sen, sen**2, -cos*sen,-sen**2],
                             [-cos**2, -cos*sen, cos**2, cos*sen],
                             [-cos*sen, -sen**2, cos*sen, sen**2]])
    
    return k_elemental


def Kglobal(MN, MC, E, A, glxn):
    Nn = MN.shape[0]        # Número de nodos
    Ne, Nnxe = MC.shape     # Ne: Número de elementos, # Número de nodos x elemento

    # out of context: el tamaño de k_elemental == glxn*nodos_x_elemento

    Kg = np.zeros([glxn*Nn, glxn*Nn])
    for e in range(Ne):
        Ee = E[e]
        Ae = A[e]
        Ke = Kelemental(MN, MC, Ee, Ae, e)
        for i in range(Nnxe):

            indices_i = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int) 
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                indices_j = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(indices_i, indices_j)]
    return Kg



