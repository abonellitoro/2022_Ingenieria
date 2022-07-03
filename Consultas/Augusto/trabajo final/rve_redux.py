import sys

# sys.path.insert(1, '../../utils')  # insert at 1, 0 is the script path (or '' in REPL)
# import fem
import gmsh
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'monospace',
        # 'weight' : 'bold',
        'size': 20}
plt.rc('figure', figsize=(16, 9))
plt.rc('font', **font)  # pass in the font dict as kwargs
plt.rc('lines', lw=2, markersize=13)


class RVE():
    def __init__(self, ff, r_f, L, lc, Es, nus,name):
        '''
        ff: fracción de fibra
        r_f: radio de la fibra
        L: largo
        lc: parámetro que pide el gmsh para definir el mallado
        E: módulo de Young
        nu: módulo de Poisson
        name: nombre del archivo
        '''
        self.lc = lc
        self.L = L
        self.name = name
        self.ff = ff
        self.r_f = r_f
        self.Es = Es
        self.nus = nus

        self.dimension = 3
        self.glxn = 6

        self.points = dict()
        self.lines = dict()
        self.curves = dict()
        self.surfaces = dict()
        self.fibres = dict()
        self.volume = None
        gmsh.initialize()
        gmsh.model.add(name)
        gmsh.option.setNumber("General.Terminal", 1)
        gmsh.option.setNumber("Mesh.Algorithm", 5)  # delquad, para que realice el mesh cuadrangular
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

        self.build_planar_geometry()
        # self.build_geometry()
        # self.mesh()
        # self.MN, self.MC, self.Nn, self.Ne, self.Nnxe = self.get_fem_data()
        #
        # self.get_k_global(E, self.glxn, nu=None, A=None)
        # self.set_boundary_conditions_and_get_nodes()

    def build_planar_geometry(self):
        self.points['p1'] = gmsh.model.occ.addPoint(0, 0, 0, self.lc)
        self.points['p2'] = gmsh.model.occ.addPoint(0, self.L, 0, self.lc)
        self.points['p3'] = gmsh.model.occ.addPoint(self.L, self.L, 0, self.lc)
        self.points['p4'] = gmsh.model.occ.addPoint(self.L, 0, 0, self.lc)

        # lineas
        self.lines['l1'] = gmsh.model.occ.addLine(self.points['p1'], self.points['p2'])
        self.lines['l2'] = gmsh.model.occ.addLine(self.points['p2'], self.points['p3'])
        self.lines['l3'] = gmsh.model.occ.addLine(self.points['p3'], self.points['p4'])
        self.lines['l4'] = gmsh.model.occ.addLine(self.points['p4'], self.points['p1'])

        # Armo la curva con todas las líneas involucradas
        self.curve = gmsh.model.occ.addCurveLoop([self.lines['l1'], self.lines['l2'], self.lines['l3'], self.lines['l4']])


        points_positions = np.loadtxt('lminmod_random_fiber_ff_0.4_rf_7e-06.txt')


        for i, position in enumerate(points_positions):
            x = position[0]+self.L/2
            y = position[1]+self.L/2
            # self.points['p'+str(i+4)] = gmsh.model.geo.addPoint(x, y, 0, lc*10)
            #MDF-COMMENT seguis teniendo el problema de las fibras que se te salen afuera de la superficie !
            #MDF-COMMENT con esto lya no da error el mallado.
            if ( x+self.r_f < self.L ) and ( x-self.r_f > 0 ) and ( y-self.r_f > 0 )and ( y+self.r_f < self.L ):
                self.fibres['f'+str(i)] = gmsh.model.occ.addCircle(x, y, 0, self.r_f)

        fibres_curve = []
        for fibre in list(self.fibres.values()):
            fibres_curve.append(gmsh.model.occ.addCurveLoop([fibre]))


        fibres_curve.append(self.curve)

        # self.surface = gmsh.model.occ.addPlaneSurface([self.curve])
        self.surface = gmsh.model.occ.addPlaneSurface(fibres_curve)
        #
        gmsh.model.setPhysicalName(2, self.surface, 'superficie')
        gmsh.model.occ.synchronize()

        #
        #
        gmsh.model.occ.synchronize()

    def mesh(self):
        gmsh.model.mesh.generate(dim=2)

    def run(self):
        gmsh.fltk.run()

if __name__ == '__main__':
    ff = 0.4
    r_f = 7e-6   # fiber radius [m]
    # l_min = 1e-8 * scale_factor
    delta = 50 * r_f
    L = delta
    # a_sq = L ** 2  # square area of the window taken
    lc = L/300
    name = 'rve'
    nu = 0.3
    rve = RVE(ff, r_f, L, lc, name, nu, name)
    # k_e = rve.k_elemental(1, 4)
    # print(k_e)
