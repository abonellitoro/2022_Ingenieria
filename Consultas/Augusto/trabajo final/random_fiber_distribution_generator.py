import numpy as np
import matplotlib.pyplot as plt


def random_fiber_distribution(ff, r_f, l_min, L, save=False, filename='random_fiber_ff_'):
    '''
    Algoritmo extraído de "Ge (2019) - An efficient method to generate random distribution of fibers in continuous fiber reinforced composites."

    ff: fracción de fibra
    r_f: radio de la fibra
    l_min: distancia mínima entre fibras
    a_sq: area al cuadrado
    save: opción para guardar en un txt
    filename: nombre del archivo
    '''

    a_sq = L ** 2
    N_of_fiber = int(ff * a_sq / (np.pi * r_f ** 2))

    w = -11.5 * ff ** 2 - 4.3 * ff + 8.5

    print('Número de fibras: ', N_of_fiber)

    # multiplico por la distancia que quiero trabajar considerando que random va entre 0 y 1
    positions = L * (2*np.random.rand(N_of_fiber, 2)-1)
    area = (np.pi * r_f ** 2) * N_of_fiber

    print('porcentaje de fibra en área: ', 1 - (a_sq - area) / a_sq)
    # flag = bool(area>=a_sq**2*ff)
    n = 0
    flag = False

    # while (area <= a_sq ** 2 * ff) or (flag == False):
    while flag == False:
        # if flag == True:
        #     position = tuple(np.random.rand(2))
        #     positions.append(position)
        flag = True
        print('try: ', n)
        n += 1

        for i in range(len(positions)):
            for j in range(len(positions)):
                xi = positions[i, 0]
                yi = positions[i, 1]
                xj = positions[j, 0]
                yj = positions[j, 1]
                d = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if (i != j) and (d <= 2 * r_f + l_min):
                    flag = False
                    xi = xi + (xi - xj) / np.abs(xi - xj) * w * np.random.rand(1) * L/1000
                    yi = yi + (yi - yj) / np.abs(yi - yj) * w * np.random.rand(1) * L/1000
                    positions[i, 0], positions[i, 1] = xi, yi


        positions_to_be_deleted = np.where(np.abs(positions) > L/2)
        new_positions = np.delete(positions, positions_to_be_deleted, 0)
        N_deleted_points = N_of_fiber - len(new_positions)
        print(N_deleted_points)
        positions = np.vstack((new_positions, L/2 * (2*np.random.rand(N_deleted_points, 2) - 1)))


    print('número de iteraciones: ', n)
    print(positions)
    if save:
        filename = filename + str(ff) + '_rf_' + str(r_f) + '.txt'
        np.savetxt(filename, positions)

    return positions


if __name__ == '__main__':
    ff = 0.4  # fiber fraction
    scale_factor = 1
    r_f = 7e-6 * scale_factor  # fiber radius [m]
    l_min = r_f/2 * scale_factor # 1e-8
    delta = 50 * r_f
    L = delta
    a_sq = L ** 2  # square area of the window taken

    positions = random_fiber_distribution(ff, r_f, l_min, L, save=True, filename='lminmod_random_fiber_ff_')

    fig, ax = plt.subplots(figsize=(7, 7))
    for position in positions:
        circle = plt.Circle(position, r_f, color='k')
        ax.add_patch(circle)

    # ax.set_aspect('equal')
    ax.set_xlim((-L / 2, L / 2))
    ax.set_ylim((-L / 2, L / 2))
    # ax.set_xlim((-L, L))
    # ax.set_ylim((-L, L))
    plt.show()
