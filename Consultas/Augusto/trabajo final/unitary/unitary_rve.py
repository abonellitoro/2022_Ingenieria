import itertools

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
T=-1

name = 'prueba'
gmsh.initialize()
gmsh.model.add(name)

# # para hacer que los elementos sean cuadrados
# gmsh.option.setNumber("General.Terminal", 1)
# gmsh.option.setNumber("Mesh.Algorithm", 5)  # delquad, para que realice el mesh cuadrangular
# gmsh.option.setNumber("Mesh.RecombineAll", 1)


cube = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
c_bl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, r_f)
c_ul = gmsh.model.occ.addCylinder(0, L, 0, 0, 0, L, r_f)
c_br = gmsh.model.occ.addCylinder(L, 0, 0, 0, 0, L, r_f)
c_ur = gmsh.model.occ.addCylinder(L, L, 0, 0, 0, L, r_f)
c_center = gmsh.model.occ.addCylinder(L / 2, L / 2, 0, 0, 0, L, r_f)
print('Entidades: ', gmsh.model.getEntities(3))
fiber_tags = [c_bl, c_ul, c_br, c_ur, c_center]
old_tags = {cube, c_bl, c_ul, c_br, c_ur, c_center}

volume = gmsh.model.occ.intersect([(3, cube)], [(3, c_ul), (3, c_ur), (3, c_bl), (3, c_br), (3, c_center)],
                                  removeObject=False, removeTool=True)
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()
print('Entidades: ', gmsh.model.getEntities(3))

new_tags = {entity[1] for entity in gmsh.model.getEntities(3)}
matrix_tag = list(new_tags - old_tags)

# volume_tags = [volume_element[1] for volume_element in volume[0]]
# volume_pg = gmsh.model.addPhysicalGroup(3, volume_tags)
# gmsh.model.setPhysicalName(3, volume_pg, 'volume')
gmsh.model.occ.synchronize()
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

# Defino los Physical Groups
bottom_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in
                                                          bottom_face_entities])  # , name='bottom_face')
gmsh.model.setPhysicalName(2, bottom_face_PG, 'bottom_face')

top_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in top_face_entities])
gmsh.model.setPhysicalName(2, bottom_face_PG, 'top_face')

fibers_PG = gmsh.model.addPhysicalGroup(dim=3, tags=[c_bl, c_ul, c_br, c_ur, c_center])
gmsh.model.setPhysicalName(3, fibers_PG, 'fibers')

matrix_PG = gmsh.model.addPhysicalGroup(dim=3, tags=matrix_tag)
gmsh.model.setPhysicalName(3, matrix_PG, 'matrix')

cube_PG = gmsh.model.addPhysicalGroup(dim=3, tags=[cube])
gmsh.model.setPhysicalName(3, cube_PG, 'cube_volume')

gmsh.option.setNumber("Mesh.MeshSizeMin", 1.2)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 4.4)
gmsh.model.mesh.generate(dim=3)
# gmsh.model.mesh.refine()
# gmsh.model.mesh.refine()
# gmsh.model.mesh.refine()

gmsh.model.occ.synchronize()

MN, MC, Nn, Ne, Nnxe, nodes_info, etags = get_fem_data(dimension=3)

matrix_entities = gmsh.model.getEntitiesForPhysicalGroup(3, matrix_PG)
matrix_elements_list = [gmsh.model.mesh.getElements(3, matrix_entity) for matrix_entity in matrix_entities]
fibers_entities = gmsh.model.getEntitiesForPhysicalGroup(3, fibers_PG)
fibers_elements_list = [gmsh.model.mesh.getElements(3, fiber_entity) for fiber_entity in fibers_entities]

# aux =
fibers_elements = np.concatenate([fiber_element[1][0] for fiber_element in fibers_elements_list])
matrix_elements = matrix_elements_list[0][1][0]
# getElements returns `elementTypes', `elementTags', `nodeTags'.

E_fiber = np.array([230e9] * len(fibers_elements))
E_matrix = np.array([2e9] * len(matrix_elements))
Es = np.concatenate((E_fiber, E_matrix))
glxn = 3
nu = [.28] * Ne  # 0.26 a 0.28 I. Krucinska and T. Stypka, Compos. Sci. Technol. 41, 1-12 (1991).
K, B, D = get_k_global(MN, MC, Es, glxn, nu=nu, A=None)
# gmsh.model.setColor([(3, matrix_tag[0])], 100, 100, 240)
# gmsh.model.setColor([(3, fiber_tag) for fiber_tag in fiber_tags], 240, 80, 80)

# Construyo las condiciones de contorno
bottom_face_labels, bottom_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, bottom_face_PG)
top_face_labels, top_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, top_face_PG)

top_face_nodes = top_face_nodes_flatten.reshape(len(top_face_labels), dimension)

s = ((bottom_face_labels - 1) * glxn + y_direction).astype(int)
Us = np.zeros_like(s)

r = get_complementary_array(Nn * glxn, s).astype(int)

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

    M_aux = np.array([[1, MN[n1, 0], MN[n1, 2]],
                      [1, MN[n2, 0], MN[n2, 2]],
                      [1, MN[n3, 0], MN[n3, 2]]])

    A = np.abs((1 / 2) * np.linalg.det(M_aux))

    # Aplico la fuerza en la dirección y
    Fr[r == n1 * glxn + y_direction] += (T * A).astype(np.float64)/3
    Fr[r == n2 * glxn + y_direction] += (T * A).astype(np.float64)/3
    Fr[r == n3 * glxn + y_direction] += (T * A).astype(np.float64)/3

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

U3D = U.reshape(Nn, glxn)/U.max()
MNdef = MN + U3D

strains = gmsh.view.add("Desplazamientos")
# por algun motivo le faltaba sumar 1 a nodeinfo
strain_model_data = gmsh.view.addModelData(strains, 0, name, 'NodeData', nodes_info[0], U3D,
                                           numComponents=3)
gmsh.option.setNumber(f'View[{strains}].VectorType', 5)

F3D = F.reshape(Nn, glxn)

forces = gmsh.view.add('forces')
forces_model_data = gmsh.view.addModelData(forces, 0, name, 'NodeData', nodes_info[0], F3D, numComponents=3)


gmsh.option.setNumber(f'View[{forces}].VectorType', 4)
gmsh.option.setNumber(f'View[{forces}].GlyphLocation', 2)
#

gmsh.fltk.run()
    