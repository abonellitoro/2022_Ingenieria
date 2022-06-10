import numpy as np

# La función "solve" calcula los vectores de fuerzas "F" y de desplazamientos "U", empleando MEF.
def solve(K, r, Fr, s, Us):
    """
    INPUTS:
      K  = Matriz K global (relaciona los desplazamientos con las fuerzas)
      r  = Vector con los nodos con condiciones de vínculo de fuerza
      Fr = Vector con las fuerzas en cada nodo del vector 'r'
      s  = Vector con los nodos con condiciones de vínculo de desplazamiento
      Us = Vector con los desplazamientos en cada nodo del vector 's'
    OUTPUTS:
      F = Vector de fuerzas en cada nodo
      U = Vector de desplazamientos de cada nodo
    """
    N = np.shape(K)[1]   # Número de nodos
    F = np.zeros([N, 1]) # Vector de fuerzas "F", una fila por cada nodo.
    U = np.zeros([N, 1]) # Vector de desplazamientos "U", una fila por cada nodo.
    U[s] = Us            # Completo "U" con los desplazamientos "Us" que ya conozco.
    F[r] = Fr            # Completo "F" con las fuerzas "Fr" que ya conozco.
    # Teniendo en cuenta que los desplazamientos "Us" y las fuerzas "Fr" son conocidas, buscaré identificar el subsistema de
    # ecuaciones tal que "Ur" sean las incógnitas que podría calcular con "Fr". De aquí surge la partición de la matriz 
    # global "K" en las submatrices "Kr" (MATRIZ REDUCIDA, asociada a desplazamientos incógnitas "Ur") y "Kv" (MATRIZ DE
    # VÍNCULOS, asociada a desplazamientos conocidos "Us").
    Kr = K[np.ix_(r, r)]
    Kv = K[np.ix_(r, s)]    
    U[r] = np.linalg.solve(Kr, F[r]-Kv.dot(U[s])) # Obtengo los desplazamientos incógnita "Ur", a partir de "Fr", "Us" y "Kv".
    F[s] = K[s, :].dot(U) # Tengo todos los desplazamientos "U" conocidos. Calculo las fuerzas "Fs" incógnitas que me faltan.
    return F, U

# La función "Kelemental" calcula la matriz elemental "Ke" del elemento "e".
def Kelemental(MN, MC, Ee, Ae, e):
    """
    INPUTS:
      MN = Matriz de nodos
      MC = Matriz de conectividad
      Ee = Módulo elástico del elemento
      Ae = Sección del elemento
      e  = Número de elemento
    OUTPUTS:
      Ke = Matriz K elemental
    """
    nodo1 = MC[e, 0]   # Primer nodo que conforma al elemento "e".
    nodo2 = MC[e, 1]   # Segundo nodo que conforma al elemento "e".
    Lx = MN[nodo2, 0]-MN[nodo1, 0]   # Longitud en eje "x".
    Ly = MN[nodo2, 1]-MN[nodo1, 1]   # Longitud en eje "y".
    L = np.sqrt(Lx**2+Ly**2)         # Longitud del elemento "e" (calculé la norma).
    phi = np.arctan2(Ly, Lx)         # Ángulo de inclinación del elemento "e".
    cos = np.cos(phi)
    sin = np.sin(phi)
    Ke = (Ee*Ae/L)*np.array([[cos**2, cos*sin, -cos**2, -cos*sin],
                             [cos*sin, sin**2, -cos*sin, -sin**2],
                             [-cos**2, -cos*sin, cos**2, cos*sin],
                             [-cos*sin, -sin**2, cos*sin, sin**2]])
    Ke[np.abs(Ke/Ke.max()) < 1e-15] = 0   # Para que me aparezca "0" en lugar de exponentes "1e-34", por ejemplo.
    return Ke

# La función "Kglobal" calcula la matriz global "K".
def Kglobal(MN, MC, E, A, glxn):
    """
    INPUTS:
      MN   = Matriz de nodos
      MC   = Matriz de conectividad
      E    = Vector de módulos elásticos de cada elemento
      A    = Vector de secciones de cada elemento
      glxn = Grados de libertad por nodo
    OUTPUTS:
      Kg = Matriz K global
    """
    Nn = MN.shape[0]      # "Nn" es número de nodos.
    Ne, Nnxe = MC.shape   # "Ne" es número de elementos. "Nnxe" es número de nodos por elemento.
    Kg = np.zeros([glxn*Nn, glxn*Nn])   # Defino matriz global "Kg".
    
    archivo= 'Matrices_elementales.txt'
    with open(archivo,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matrices Elementales\n ===============')
    archivo1= 'Matriz_global.txt'
    with open(archivo1,'w') as f:   # Creo archivo desde cero, por eso uso "w".
        f.write('Matriz Global\n ===============')
    
    for e in range(Ne): 
        if glxn == 1:
            Ke = np.array([[1,-1],[-1,1]])*A*E/(MN[-1]/Ne)   # MN[-1] es la longitud "L" de la barra entera.
        elif glxn == 2:
            Ke = Kelemental(MN, MC, E[e], A[e], e)
        fe = np.abs(Ke.max()) # Factor de escala, para que los números en "Ke" se lean mejor.
        with open(archivo,'a') as f:   # Voy reescribiendo el archivo con nuevas "Ke", por eso uso "a".
            f.write(f'\nelemento {e}, fe = {fe:4e}\n')
            f.write(f'{Ke/fe}\n')
        for i in range(Nnxe):
            rangoi = np.linspace(i*glxn, (i+1)*glxn-1, Nnxe).astype(int)
            rangoni = np.linspace(MC[e, i]*glxn, (MC[e, i]+1)*glxn-1, Nnxe).astype(int)
            for j in range(Nnxe):
                rangoj = np.linspace(j*glxn, (j+1)*glxn-1, Nnxe).astype(int)
                rangonj = np.linspace(MC[e, j]*glxn, (MC[e, j]+1)*glxn-1, Nnxe).astype(int)
                Kg[np.ix_(rangoni, rangonj)] += Ke[np.ix_(rangoi, rangoj)]
    fe = np.abs(Kg.max())
    with open(archivo1,'a') as f:   # Reescribo el archivo con la matriz global "Kg" obtenida, por eso uso "a".
        f.write(f'\nMatriz Global, fe = {fe:4e}\n')
        f.write(f'{Kg/fe}\n')
    return Kg

# La función "subdiv" particiona un elemento de longitud "L" en "Ne" elementos de igual longitud.
def subdiv(E, A, L, C, Ne, glxn, Nnxe=2):
    """
    INPUTS:
      E    = Módulo de elasticidad del elemento
      A    = Sección del elemento
      L    = Longitud del elemento
      C    = Carga axial aplicada (DISTRIBUIDA)
      Ne   = Cantidad de elementos en que dividiré mi elemento original
      glxn = Grados de libertad por nodo
    OUTPUTS:
      F = Vector de fuerzas en cada nodo
      U = Vector de desplazamientos de cada nodo
      R_emp = Reacción del empotramiento sobre el último nodo
      sigma = Vector de tensiones en cada barra
      f = Vector que contiene fuerza aplicada sobre cada nodo DEBIDA a CARGA DISTRIBUIDA
      Kg = Matriz global
    """
    
    MN = np.linspace(0,L,Ne+1).reshape([-1,1])         # Matriz de nodos.
    Nn = MN.shape[0]                                   # "Nn" es número de nodos.
    
    MC = np.array([[i, i+1] for i in range(Ne)])       # Matriz de conectividad
    Ne, Nnxe = MC.shape                                # "Ne" es número de elementos. "Nnxe" es número de nodos por elemento.
    Le = L/Ne                                          # Longitud de los elementos resultantes.
   
    K = Kglobal(MN, MC, E, A, glxn)                    # Matriz global
    
    # Elemento "0"
    FT = 0.5*C*(Le**2)   # Fuerza total "FT" actuante sobre el elemento "0".
    # Distrubuyo la carga distribuida "T(x)" en los nodos "0" y "1".
    f0 = FT/3     # - Al nodo "0" le corresponde "f0 = FT/3".
    f1 = 2*FT/3   # - Al nodo "1" le corresponde "f1 = 2*FT/3".
    
    f = np.zeros([Ne+1])
    f[0] = f0
    f[1] = f1
    
    # RESTO de elementos
    for i in range(1,Ne): 
        # Contribución debida a RECTÁNGULOS.
        # - "Ne=2" hace que a lo sumo "i=1", entonces "1" rectángulo de área "2*FT" entrega "FT" a cada nodo.
        # - "Ne=3" hace que a lo sumo "i=2", entonces "2" rectángulos de área "2*FT" entregan "2*FT" a cada nodo.
        # - "Ne=4" hace que a lo sumo "i=3", entonces "3" rectángulos de área "2*FT" entregan "3*FT" a cada nodo.
        Ft = FT*i   
        # Fuerza en nodos de la barra i = suma de todos los rectangulos (Ft*i) + carga de la primer barra(triangulito) (Ft)
        f[i] += f0+Ft
        f[i+1] += f1+Ft
    
    s = np.array([Ne])   # Vector "s" que contiene los nodos con condiciones de vínculo en desplazamiento.
    Us = [[0]]           # Vector "Us" con los valores de las condiciones de vínculo. EMPOTRAMIENTO.
    r = np.array([i for i in range(Nn*glxn) if i not in s])   # Vector "r" que contiene los nodos con condiciones de vínculo en fuerza.
    Fr = np.array([[f[i]] for i in range(Ne)])   # Vector "Fr" con los valores de las condiciones de vínculo. CARGAS DISTRIBUIDAS.
    
    F, U = solve(K, r, Fr, s, Us)
    
    R_emp = F[-1] - f[-1]   # Calculo la reacción "R_emp" del empotramiento sobre el último nodo "i = -1".
    
    eps = np.zeros([Ne,1])
    sigma = np.zeros([Ne,1])
    for i in range(Ne):
        eps[i] = (U[i+1]-U[i])/(Le)
        sigma[i] = eps[i]*E
    
    return F, U, R_emp, sigma, f, K