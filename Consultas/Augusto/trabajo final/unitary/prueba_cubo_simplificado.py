import gmsh
import numpy as np
from utils_tf import get_fem_data, get_k_global, get_complementary_array, solve

x_direction, y_direction, z_direction = (0, 1, 2)

ff = 0.4
r_f = 7e-6  # fiber radius [m]
l_min = 1e-6

H = 2 * r_f + l_min
L = 2 * np.sqrt(H ** 2 / 2)
lc = L / 10
dimension = 3

name = 'prueba'
gmsh.initialize()
gmsh.model.add(name)

cube = gmsh.model.occ.addBox(0, 0, 0, L, L, L)

dr = l_min

# Dirección Z
rear_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, -dr, -dr, L + dr, L + dr, dr, dim=2)
front_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, -dr, L - dr, L + dr, L + dr, L + dr, dim=2)
# Dirección Y
bottom_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, -dr, -dr, L + dr, dr, L + dr, dim=2)
top_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, L - dr, -dr, L + dr, L + dr, L + dr, dim=2)
# Dirección X
right_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, - dr, -dr, dr, L + dr, L + dr, dim=2)
left_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(L - dr, -dr, -dr, L + dr, L + dr, L + dr, dim=2)

bottom_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in
                                                          bottom_face_entities])  # , name='bottom_face')

gmsh.model.setPhysicalName(2, bottom_face_PG, 'bottom_face')

top_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in top_face_entities])
gmsh.model.setPhysicalName(2, bottom_face_PG, 'top_face')

gmsh.option.setNumber("Mesh.MeshSizeMin", 1.2)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 4.4)
gmsh.model.occ.synchronize()

gmsh.model.mesh.generate(dim=3)

gmsh.model.occ.synchronize()

MN, MC, Nn, Ne, Nnxe, nodes_info, etags = get_fem_data(dimension=3)

Es = [230e9]*Ne
glxn = 3
nu = [.28] * Ne  # 0.26 a 0.28 I. Krucinska and T. Stypka, Compos. Sci. Technol. 41, 1-12 (1991).
K, B, D = get_k_global(MN, MC, Es, glxn, nu=nu, A=None)

# Construyo las condiciones de contorno
bottom_face_labels, bottom_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, bottom_face_PG)
top_face_labels, top_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, top_face_PG)

top_face_nodes = top_face_nodes_flatten.reshape(len(top_face_labels), dimension)

s = ((bottom_face_labels - 1) * glxn + y_direction).astype(int)
Us = np.zeros_like(s)

r = get_complementary_array(Nn * glxn, s).astype(int)

T = -1000  # psi

elementTypes_array = np.array([])
elementTags_array = np.array([])
nodeTags_array = np.array([])

for entity in top_face_entities:
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(*entity)
    elementTypes_array = np.append(elementTypes_array, elementTypes)
    elementTags_array = np.append(elementTags_array, elementTags)
    nodeTags_array = np.append(nodeTags_array, nodeTags)

Ne_stress = len(elementTags_array)
MC_stress = nodeTags_array.reshape(Ne_stress, glxn) - 1
Fr = np.zeros_like(r).astype(np.float64)
for e_ in range(Ne_stress):
    n1 = MC_stress[e_, 0].astype(int)
    n2 = MC_stress[e_, 1].astype(int)
    n3 = MC_stress[e_, 2].astype(int)

    M_aux = np.array([[1, MN[n1, 0], MN[n1, 1]],
                      [1, MN[n2, 0], MN[n2, 1]],
                      [1, MN[n3, 0], MN[n3, 1]]])

    A = np.abs((1 / 2) * np.linalg.det(M_aux))

    # Aplico la fuerza en la dirección y
    Fr[np.where(r == n1 * glxn)[0][0] + y_direction] += (T * A).astype(np.float64)/3
    Fr[np.where(r == n2 * glxn)[0][0] + y_direction] += (T * A).astype(np.float64)/3
    Fr[np.where(r == n3 * glxn)[0][0] + y_direction] += (T * A).astype(np.float64)/3

U, F = solve(K, s, r, Us, Fr)

sig = {}
d = {}
for e in range(Ne):
    nodo = MC[e, :].astype(int)
    d[e] = np.array([U[nodo[0] * glxn], U[nodo[0] * glxn + 1], U[nodo[0] * glxn + 2],
                     U[nodo[1] * glxn], U[nodo[1] * glxn + 1], U[nodo[1] * glxn + 2],
                     U[nodo[2] * glxn], U[nodo[2] * glxn + 1], U[nodo[2] * glxn + 2],
                     U[nodo[3] * glxn], U[nodo[3] * glxn + 1], U[nodo[3] * glxn + 2]]).reshape([-1, 1])
    sig[e] = D[e].dot(B[e].dot(d[e]))

U3D = U.reshape(Nn, glxn)
MNdef = MN + U3D * 1e4

strains = gmsh.view.add("Desplazamientos")
# por algun motivo le faltaba sumar 1 a nodeinfo
strain_model_data = gmsh.view.addModelData(strains, 0, name, 'NodeData', nodes_info[0] + 1, U3D,
                                           numComponents=3)
gmsh.option.setNumber(f'View[{strains}].VectorType', 5)

F3D = F.reshape(Nn, glxn)

forces = gmsh.view.add('forces')
forces_model_data = gmsh.view.addModelData(forces, 0, name, 'NodeData', nodes_info[0] + 1, F3D, numComponents=3)
gmsh.option.setNumber(f'View[{forces}].VectorType', 4)
gmsh.option.setNumber(f'View[{forces}].GlyphLocation', 2)

gmsh.fltk.run()
