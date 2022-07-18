import gmsh
import numpy as np

from utils_tf import get_fem_data, get_k_global, solve, calculate_boundary_conditions_controlled_by_displacement, \
    calculate_boundary_conditions_controlled_by_force, addView

x_direction, y_direction, z_direction = (0, 1, 2)
directions = (x_direction, y_direction, z_direction)
ff = 0.6
r_f = 7e-6  # fiber radius [m]
l_min = 1e-6

H = 2 * r_f + l_min
L = 2 * np.sqrt(H ** 2 / 2)

a1 = a2 = a3 = L

dimension = 3
T = -1

name = 'prueba'
gmsh.initialize()
gmsh.model.add(name)

# # para hacer que los elementos sean cuadrados
gmsh.option.setNumber("General.Terminal", 1)
# gmsh.option.setNumber("Mesh.Algorithm", 5)  # delquad, para que realice el mesh cuadrangular
# gmsh.option.setNumber("Mesh.RecombineAll", 1)


cube = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
c_bl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, r_f)
c_ul = gmsh.model.occ.addCylinder(0, L, 0, 0, 0, L, r_f)
c_br = gmsh.model.occ.addCylinder(L, 0, 0, 0, 0, L, r_f)
c_ur = gmsh.model.occ.addCylinder(L, L, 0, 0, 0, L, r_f)
c_center = gmsh.model.occ.addCylinder(L / 2, L / 2, 0, 0, 0, L, r_f)
fiber_tags = [c_bl, c_ul, c_br, c_ur, c_center]
old_tags = {cube, c_bl, c_ul, c_br, c_ur, c_center}

volume = gmsh.model.occ.intersect([(3, cube)], [(3, c_ul), (3, c_ur), (3, c_bl), (3, c_br), (3, c_center)],
                                  removeObject=False, removeTool=True)
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()

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
bottom_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in bottom_face_entities],
                                             name='bottom_face')  # , name='bottom_face')
top_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in top_face_entities], name='top_face')
rear_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in rear_face_entities], name='rear_face')
front_face_PG = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in front_face_entities],
                                            name='front_face')

fibers_PG = gmsh.model.addPhysicalGroup(dim=3, tags=[c_bl, c_ul, c_br, c_ur, c_center])
gmsh.model.setPhysicalName(3, fibers_PG, 'fibers')

matrix_PG = gmsh.model.addPhysicalGroup(dim=3, tags=matrix_tag)
gmsh.model.setPhysicalName(3, matrix_PG, 'matrix')

cube_PG = gmsh.model.addPhysicalGroup(dim=3, tags=[cube])
gmsh.model.setPhysicalName(3, cube_PG, 'cube_volume')

gmsh.option.setNumber("Mesh.MeshSizeMin", 1.2)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 4.4)
gmsh.model.mesh.generate(dim=3)
gmsh.model.mesh.refine()
gmsh.model.mesh.refine()
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

glxn = 3

# Valores extraídos del Barbero
E_fiber = np.array([241e9] * len(fibers_elements))
nu_fiber = np.array([0.2] * len(fibers_elements))
E_matrix = np.array([3.12e9] * len(matrix_elements))
nu_matrix = np.array([0.38] * len(matrix_elements))

Es = np.concatenate((E_fiber, E_matrix))
nus = np.concatenate((nu_fiber, nu_matrix))
# nu = [.28] * Ne  # 0.26 a 0.28 I. Krucinska and T. Stypka, Compos. Sci. Technol. 41, 1-12 (1991).
K, B, D = get_k_global(MN, MC, Es, glxn, nu=nus, A=None)
gmsh.model.setColor([(3, matrix_tag[0])], 100, 100, 240)
gmsh.model.setColor([(3, fiber_tag) for fiber_tag in fiber_tags], 240, 80, 80)

# Construyo las condiciones de contorno
bottom_face_labels, bottom_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, bottom_face_PG)
top_face_labels, top_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, top_face_PG)
rear_face_labels, rear_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, rear_face_PG)
front_face_labels, front_face_nodes_flatten = gmsh.model.mesh.getNodesForPhysicalGroup(2, front_face_PG)

top_face_nodes = top_face_nodes_flatten.reshape(len(top_face_labels), dimension)

# Aplico el desplazamiento en el sentido ortogonal a la fibra
s, Us, r, Fr = calculate_boundary_conditions_controlled_by_displacement(static_face=bottom_face_labels,
                                                                        displaced_face=top_face_labels,
                                                                        displacement_direction=y_direction,
                                                                        displacement_amount=-L,
                                                                        glxn=glxn,
                                                                        MN=MN)

# Aplico el desplazamiento en el sentido de la fibra
# s, Us, r, Fr = calculate_boundary_conditions_controlled_by_displacement(static_face= rear_face_labels,
#                                                                         displaced_face=front_face_labels,
#                                                                         displacement_direction=z_direction,
#                                                                         displacement_amount=l_min,
#                                                                         glxn=glxn,
#                                                                         MN=MN)

# Aplico el esfuerzo en el sentido ortoganal a la fibra
# s, Us, r, Fr = calculate_boundary_conditions_controlled_by_force(static_face=bottom_face_labels,
#                                                                  stressed_face_entities=top_face_entities,
#                                                                  stress_amount=-1,
#                                                                  force_direction=y_direction,
#                                                                  glxn=glxn,
#                                                                  MN=MN)


# Aplico la fuerza en el sentido de la fibra
# s, Us, r, Fr = calculate_boundary_conditions_controlled_by_force(static_face=rear_face_labels,
#                                                                  stressed_face_entities=front_face_entities,
#                                                                  stress_amount=-1,
#                                                                  force_direction=z_direction,
#                                                                  glxn=glxn,
#                                                                  MN=MN)

U, F = solve(K, s, r, Us, Fr)

sig = {}
d = {}
epsilon = {}
for e in range(Ne):
    nodo = MC[e, :].astype(int)
    d[e] = np.array([U[nodo[0] * glxn], U[nodo[0] * glxn + 1], U[nodo[0] * glxn + 2],
                     U[nodo[1] * glxn], U[nodo[1] * glxn + 1], U[nodo[1] * glxn + 2],
                     U[nodo[2] * glxn], U[nodo[2] * glxn + 1], U[nodo[2] * glxn + 2],
                     U[nodo[3] * glxn], U[nodo[3] * glxn + 1], U[nodo[3] * glxn + 2]]).reshape([-1, 1])
    epsilon[e] = B[e].dot(d[e])
    sig[e] = D[e].dot(epsilon[e])

U3D = U.reshape(Nn, glxn)
MNdef = U3D
addView(MNdef, name, 'Desplazamiento', nodes_info[0], 'NodeData', vectorType=5)  # para que mire desplazamiento
# strain_model_data = gmsh.view.addModelData(displacement, 0, name, 'NodeData', nodes_info[0], U3D, numComponents=3)

F3D = F.reshape(Nn, glxn)
addView(F3D, name, 'Fuerzas', nodes_info[0], 'NodeData')

epsilon_ = np.hstack(list(epsilon.values())).transpose()
epsilonX = epsilon_[:, 0].reshape([-1, 1])
epsilonY = epsilon_[:, 1].reshape([-1, 1])
epsilonZ = epsilon_[:, 2].reshape([-1, 1])
addView(epsilonX, name, 'Deformaciones en X', etags, 'ElementData')
addView(epsilonY, name, 'Deformaciones en Y', etags, 'ElementData')
addView(epsilonZ, name, 'Deformaciones en Z', etags, 'ElementData')

sig_xytau = np.hstack(list(sig.values()))
sigX = sig_xytau[0, :].reshape([-1, 1])
sigY = sig_xytau[1, :].reshape([-1, 1])
sigZ = sig_xytau[2, :].reshape([-1, 1])
addView(sigX, name, 'Tensiones en X', etags, 'ElementData')
addView(sigY, name, 'Tensiones en Y', etags, 'ElementData')
addView(sigZ, name, 'Tensiones en Z', etags, 'ElementData')

gmsh.fltk.run()
