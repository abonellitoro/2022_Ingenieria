import gmsh
import numpy as np
from utils_tf import get_fem_data, get_k_global

ff = 0.4
r_f = 7e-6  # fiber radius [m]
l_min = 1e-6

H = 2 * r_f + l_min
L = 2 * np.sqrt(H ** 2 / 2)
lc = L / 10

gmsh.initialize()
gmsh.model.add('prueba')

# # para hacer que los elementos sean cuadrados
# gmsh.option.setNumber("General.Terminal", 1)
# gmsh.option.setNumber("Mesh.Algorithm", 5)  # delquad, para que realice el mesh cuadrangular
# gmsh.option.setNumber("Mesh.RecombineAll", 1)

points = dict()
lines = dict()

cube = gmsh.model.occ.addBox(0, 0, 0, L, L, L)
c_bl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, L, r_f)
c_ul = gmsh.model.occ.addCylinder(0, L, 0, 0, 0, L, r_f)
c_br = gmsh.model.occ.addCylinder(L, 0, 0, 0, 0, L, r_f)
c_ur = gmsh.model.occ.addCylinder(L, L, 0, 0, 0, L, r_f)
c_center = gmsh.model.occ.addCylinder(L / 2, L / 2, 0, 0, 0, L, r_f)
print('Entidades: ', gmsh.model.getEntities(3))
old_tags = {cube, c_bl, c_ul, c_br, c_ur, c_center}

volume = gmsh.model.occ.intersect([(3, cube)], [(3, c_ul), (3, c_ur), (3, c_bl), (3, c_br), (3, c_center)], removeObject=False, removeTool=True)
gmsh.model.occ.removeAllDuplicates()
gmsh.model.occ.synchronize()
print('Entidades: ', gmsh.model.getEntities(3))

new_tags = {entity[1] for entity in gmsh.model.getEntities(3)}
matrix_tag = list(new_tags-old_tags)

# volume_tags = [volume_element[1] for volume_element in volume[0]]
# volume_pg = gmsh.model.addPhysicalGroup(3, volume_tags)
# gmsh.model.setPhysicalName(3, volume_pg, 'volume')
gmsh.model.occ.synchronize()
dr = l_min

rear_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, -dr, -dr, L+ dr, L + dr, dr, dim=2)
front_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, -dr, L-dr, L+ dr, L + dr, L+dr, dim=2)
bottom_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, -dr, -dr, L + dr, dr, L + dr, dim=2)
top_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, L - dr, -dr, L + dr, L + dr, L + dr, dim=2)
right_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(-dr, - dr, -dr, dr, L + dr, L + dr, dim=2)
left_face_entities = gmsh.model.occ.getEntitiesInBoundingBox(L-dr, -dr, -dr, L + dr, L + dr, L + dr, dim=2)


#MDF-COMMENT bottom_face = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in bottom_face_entities], name='bottom_face')
bottom_face = gmsh.model.addPhysicalGroup(dim=2, tags=[entity[1] for entity in bottom_face_entities])
gmsh.model.setPhysicalName(2, bottom_face,'bottom_face')
#top_face = gmsh.model.addPhysicalGroup(dim=2, tags=top_face_entities)
#gmsh.model.setPhysicalName(2, top_face,'top_face')
#MDF-COMMENT fibers_PG = gmsh.model.addPhysicalGroup(dim=3, tags=[c_bl, c_ul, c_br, c_ur, c_center], name='fibers')
fibers_PG = gmsh.model.addPhysicalGroup(dim=3, tags=[c_bl, c_ul, c_br, c_ur, c_center]) 
gmsh.model.setPhysicalName(3, fibers_PG, 'fibers')
#MDF-COMMENTmatrix_PG = gmsh.model.addPhysicalGroup(dim=3, tags=matrix_tag, name='matrix')
matrix_PG = gmsh.model.addPhysicalGroup(dim=3, tags=matrix_tag) #MDF-COMMENT, name='matrix')
gmsh.model.setPhysicalName(3, matrix_PG, 'matrix')

print('cara trasera: ', rear_face_entities)
print('cara frontal: ', front_face_entities)
print('cara inferior', bottom_face_entities)
print('cara superior: ', top_face_entities)
print('cara izquierda: ', right_face_entities)
print('cara derecha: ', left_face_entities)
gmsh.option.setNumber("Mesh.MeshSizeMin", 1.2)
# gmsh.option.setNumber("Mesh.MeshSizeMax", 4.4)
gmsh.model.mesh.generate(dim=3)
gmsh.model.mesh.refine()
gmsh.model.mesh.refine()
# gmsh.model.mesh.refine()

gmsh.model.occ.synchronize()

MN, MC, Nn, Ne, Nnxe = get_fem_data(dimension=3)

lower_face_nodes_labels, lower_face_coordinates = gmsh.model.mesh.getNodesForPhysicalGroup(2, bottom_face)
# upper_face_nodes_labels, upper_face_coordinates = gmsh.model.mesh.getNodesForPhysicalGroup(2, top_face)

matrix_entities = gmsh.model.getEntitiesForPhysicalGroup(3, matrix_PG)
matrix_elements_list = [gmsh.model.mesh.getElements(3, matrix_entity) for matrix_entity in matrix_entities]
fibers_entities = gmsh.model.getEntitiesForPhysicalGroup(3, fibers_PG)
fibers_elements_list = [gmsh.model.mesh.getElements(3, fiber_entity) for fiber_entity in fibers_entities]

# aux =
fibers_elements = np.concatenate([fiber_element[1][0] for fiber_element in fibers_elements_list])
matrix_elements = matrix_elements_list[0][1][0]
# getElements returns `elementTypes', `elementTags', `nodeTags'.

E_fiber = np.array([230e9]*len(fibers_elements))
E_matrix = np.array([2e9]*len(matrix_elements))
Es = np.concatenate((E_fiber, E_matrix))
#MDF-COMMENT glxn = 6 #MDF-COMMENT por qué 6 ? no dería 3?
glxn = 3
nu = [.28]*Ne #0.26 a 0.28 I. Krucinska and T. Stypka, Compos. Sci. Technol. 41, 1-12 (1991).
get_k_global(MN, MC, Es, glxn, nu=nu, A=None) # todo arreglar

gmsh.fltk.run()
