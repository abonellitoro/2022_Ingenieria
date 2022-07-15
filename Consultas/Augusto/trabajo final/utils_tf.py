import gmsh
import numpy as np
import copy

def get_fem_data(dimension):
    nodes_info = gmsh.model.mesh.get_nodes()
    Nn = nodes_info[0].shape[0]  # Número de nodos
    MN = nodes_info[1].reshape(Nn, dimension)  # Matriz de nodos

    # como estoy en una geometría 3D, utiliza un tetrahedro de 4 nodos
    # http://www.manpagez.com/info/gmsh/gmsh-2.2.6/gmsh_63.php
    etags, flattenMC = gmsh.model.mesh.get_elements_by_type(4)
    Nnxe = int(len(flattenMC) / len(etags))
    Ne = etags.shape[0]

    # etags son las etiquetas de los elmementos, ya que gmsh contiene varios tipos de elementos en simultáneo
    MC = flattenMC.reshape(Ne, Nnxe).astype(int)
    MC = MC - np.ones(MC.shape).astype(int)  # le resto 1 para que respete los índices que utiliza python

    return MN, MC, Nn, Ne, Nnxe, nodes_info, etags


def get_constitutive_matrix(E, nu):
    D = np.zeros([6, 6])
    D[0:3, 0:3] = (1 - nu) * np.eye(3)
    D[3:, 3:] = np.eye(3) * (1 - 2 * nu) / 2
    D[0, 1:3] = nu
    D[1:3, 0] = nu
    D[1, 2] = nu
    D[2, 1] = nu
    D = E / ((1 + nu) * (1 - 2 * nu)) * D
    return D

def get_constitutive_matrix_for_composites():
    E_0 = 3.12  # GPa
    nu_m = 0.38
    E_m = copy.copy(E_0)

    lm_0 = E_m * nu_m / ((1 + nu_m) * (1 - 2 * nu_m))
    mu_0 = E_m / (2 * (1 + nu_m))
    nu_0 = nu_m

    ###Fiber: isotropic
    E_f = 241  # GPa
    nu_f = 0.2

    lm_1 = nu_f * E_f / ((1 + nu_f) * (1 - 2 * nu_f))
    mu_1 = E_f / (2 * (1 + nu_f))
    nu_1 = nu_f

    ## a	b	c	g

    a = mu_1 - mu_0 - 2 * mu_1 * nu_m + 2 * mu_0 * nu_f
    b = -mu_0 * nu_m + mu_1 * nu_f + 2 * mu_0 * nu_m * nu_f - 2 * mu_1 * nu_m * nu_f
    c = (mu_0 - mu_1) * (
                -mu_0 + mu_1 - mu_0 * nu_m - 2 * mu_1 * nu_m + 2 * mu_0 * nu_f + mu_1 * nu_f + 2 * mu_0 * nu_m * nu_f - 2 * mu_1 * nu_m * nu_f)
    g = 2 - 2 * nu_m

    V_f = 0.6

    ##S_3	S_6	S_7
    S_3 = 0.49247 - 0.47603 * V_f - 0.02748 * V_f ** 2
    S_6 = 0.36844 - 0.14944 * V_f - 0.27152 * V_f ** 2
    S_7 = 0.12346 - 0.32035 * V_f + 0.23517 * V_f ** 2

    ##D_	C_11*	C_12*	C_23*	C_22*	C_44*	C_66*
    D_ = (a * S_3 ** 2) / (2 * mu_0 ** 2 * c) - (a * S_6 * S_3) / (mu_0 ** 2 * g * c) + a * (S_6 ** 2 - S_7 ** 2) / (
                2 * mu_0 ** 2 * g ** 2 * c) + S_3 * (b ** 2 - a ** 2) / (2 * mu_0 * c ** 2) + (
                     S_6 * (a ** 2 - b ** 2) + S_7 * (a * b + b ** 2)) / (2 * mu_0 * g * c ** 2) + (
                     a ** 3 - 2 * b ** 3 - 3 * a * b ** 2) / (8 * c ** 3)
    C_11 = lm_0 + 2 * mu_0 - V_f * (
                (S_3 ** 2) / (mu_0 ** 2) - 2 * S_6 * S_3 / (mu_0 ** 2 * g) - a * S_3 / (mu_0 * c) + (
                    S_6 ** 2 - S_7 ** 2) / (mu_0 ** 2 * g ** 2) + (a * S_6 + S_7 * b) / (mu_0 * g * c) + (
                            a ** 2 - b ** 2) / (4 * c ** 2)) / D_
    C_12 = lm_0 + ((S_3 / (2 * c * mu_0)) - ((S_6 - S_7) / (2 * mu_0 * c * g)) - (a + b) / (4 * c ** 2)) * b * V_f / D_
    C_23 = lm_0 + V_f * ((a * S_7) / (2 * mu_0 * g * c) - (b * a + b ** 2) / (4 * c ** 2)) / D_
    C_22 = lm_0 + 2 * mu_0 - V_f * (
                (-a * S_3) / (2 * mu_0 * c) + (a * S_6) / (2 * mu_0 * g * c) + (a ** 2 - b ** 2) / (4 * c ** 2)) / D_
    C_44 = mu_0 - V_f / ((-2 * S_3) / mu_0 + 1 / (mu_0 - mu_1) + 4 * S_7 / (mu_0 * (2 - 2 * nu_0)))
    C_66 = mu_0 - V_f / ((-S_3 / mu_0) + 1 / (mu_0 - mu_1))


    D = np.zeros([6, 6])
    C_ii = [C_11, C_22, C_22, 1/2*(C_22-C_23), C_66, C_66]
    for i in range(len(D)):
        D[i,i] = C_ii[i]
    D[0, 1:3] = C_12
    D[1:3, 0] = C_12
    D[1, 2] = C_23
    D[2, 1] = C_23


    return D


def get_k_elemental(element, MN, E, nu, glxn):
    nodos_del_elemento = element
    M = np.array([[1, MN[nodos_del_elemento[0], 0], MN[nodos_del_elemento[0], 1], MN[nodos_del_elemento[0], 2]],
                  [1, MN[nodos_del_elemento[1], 0], MN[nodos_del_elemento[1], 1], MN[nodos_del_elemento[1], 2]],
                  [1, MN[nodos_del_elemento[2], 0], MN[nodos_del_elemento[2], 1], MN[nodos_del_elemento[2], 2]],
                  [1, MN[nodos_del_elemento[3], 0], MN[nodos_del_elemento[3], 1], MN[nodos_del_elemento[3], 2]]])

    # print(np.linalg.det(M))
    alpha_1 = np.linalg.det(np.delete(np.delete(M, 0, axis=0), 0, axis=1))
    beta_1 = -np.linalg.det(np.delete(np.delete(M, 0, axis=0), 1, axis=1))
    gamma_1 = np.linalg.det(np.delete(np.delete(M, 0, axis=0), 2, axis=1))
    delta_1 = -np.linalg.det(np.delete(np.delete(M, 0, axis=0), 3, axis=1))

    alpha_2 = -np.linalg.det(np.delete(np.delete(M, 1, axis=0), 0, axis=1))
    beta_2 = np.linalg.det(np.delete(np.delete(M, 1, axis=0), 1, axis=1))
    gamma_2 = -np.linalg.det(np.delete(np.delete(M, 1, axis=0), 2, axis=1))
    delta_2 = np.linalg.det(np.delete(np.delete(M, 1, axis=0), 3, axis=1))

    alpha_3 = np.linalg.det(np.delete(np.delete(M, 2, axis=0), 0, axis=1))
    beta_3 = -np.linalg.det(np.delete(np.delete(M, 2, axis=0), 1, axis=1))
    gamma_3 = np.linalg.det(np.delete(np.delete(M, 2, axis=0), 2, axis=1))
    delta_3 = -np.linalg.det(np.delete(np.delete(M, 2, axis=0), 3, axis=1))

    alpha_4 = -np.linalg.det(np.delete(np.delete(M, 3, axis=0), 0, axis=1))
    beta_4 = np.linalg.det(np.delete(np.delete(M, 3, axis=0), 1, axis=1))
    gamma_4 = -np.linalg.det(np.delete(np.delete(M, 3, axis=0), 2, axis=1))
    delta_4 = np.linalg.det(np.delete(np.delete(M, 3, axis=0), 3, axis=1))

    # print(np.array([[alpha_1, beta_1, gamma_1, delta_1],
    #                 [alpha_2, beta_2, gamma_2, delta_2],
    #                 [alpha_3, beta_3, gamma_3, delta_3],
    #                 [alpha_4, beta_4, gamma_4, delta_4]]))

    B1 = np.array([[beta_1, 0, 0],
                   [0, gamma_1, 0],
                   [0, 0, delta_1],
                   [gamma_1, beta_1, 0],
                   [0, delta_1, gamma_1],
                   [delta_1, 0, beta_1]])

    B2 = np.array([[beta_2, 0, 0],
                   [0, gamma_2, 0],
                   [0, 0, delta_2],
                   [gamma_2, beta_2, 0],
                   [0, delta_2, gamma_2],
                   [delta_2, 0, beta_2]])

    B3 = np.array([[beta_3, 0, 0],
                   [0, gamma_3, 0],
                   [0, 0, delta_3],
                   [gamma_3, beta_3, 0],
                   [0, delta_3, gamma_3],
                   [delta_3, 0, beta_3]])

    B4 = np.array([[beta_4, 0, 0],
                   [0, gamma_4, 0],
                   [0, 0, delta_4],
                   [gamma_4, beta_4, 0],
                   [0, delta_4, gamma_4],
                   [delta_4, 0, beta_4]])

    Ve = np.linalg.det(M) / 6
    B = (1 / (6 * Ve)) * np.hstack([B1, B2, B3, B4])
    D = get_constitutive_matrix_for_composites()
    Ke = np.transpose(B).dot(D.dot(B)) * np.abs(Ve)

    return Ke, B, D


def get_k_global(MN, MC, E, glxn, nu=None, A=None):
    Ne, Nnxe = MC.shape
    Nn = len(MN)
    '''
		Ensamblado de la matriz K.
		E: Módulo de Young
		glxn: Grados de libertado por nodo
		nu: Vector de coeficientes de Poisson
		t: Vector de espesores de los elementos
		'''

    B = []
    D = []
    # out of context: el tamaño de k_elemental == glxn*nodos_x_elemento

    Kg = np.zeros([glxn * Nn, glxn * Nn])
    for element in range(Ne):
        element_row = MC[element, :]
        Ee = E[element]
        nu_e = nu[element]
        Ke, Be, De = get_k_elemental(element_row, MN, Ee, nu_e, glxn)
        B.append(Be)
        D.append(De)

        for i in range(Nnxe):
            indices_i = np.linspace(i * glxn, (i + 1) * glxn - 1, Nnxe).astype(int)
            rangoni = np.linspace(MC[element, i] * glxn, (MC[element, i] + 1) * glxn - 1, Nnxe).astype(int)
            for j in range(Nnxe):
                indices_j = np.linspace(j * glxn, (j + 1) * glxn - 1, Nnxe).astype(int)
                rangonj = np.linspace(MC[element, j] * glxn, (MC[element, j] + 1) * glxn - 1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(indices_i, indices_j)]

    return Kg, B, D


def get_complementary_array(N, s):
    '''
	Función para obtener el complementario del vector r
	N: largo del vector f
	r: posiciones complementarias al vector s
	'''

    r = np.array([i for i in range(N) if i not in s])
    return r


def solve(k, s, r, us, fr):
    N = k.shape[1]
    f = np.zeros(N)
    u = np.zeros(N)

    u[s] = us
    f[r] = fr

    k_redux = k[np.ix_(r, r)]
    k_vinculos = k[np.ix_(r, s)]
    u[r] = np.linalg.solve(k_redux, f[r] - k_vinculos.dot(u[s]))
    f[s] = k[s, :].dot(u)

    return u, f
