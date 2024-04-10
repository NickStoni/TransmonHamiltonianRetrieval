import matplotlib.pyplot as plt
import numpy as np

from Transmon import Transmon

N = 10

N_G_LOWER = -2.5
N_G_UPPER = 2.5
N_G_RES = 0.01
N_G_RANGE = np.arange(start = N_G_LOWER, stop = N_G_UPPER + N_G_RES, step = N_G_RES)


def compute_eigenvalues(e_c, e_j):
    # Rows in eigenvalues matrix will store eigenvalues for a Hamiltonian, given a particular n_g.
    eigenvalues = np.zeros((N_G_RANGE.size, 2 * N + 1), dtype = np.float64)

    transmon = None
    for i, n_g in enumerate(N_G_RANGE):
        # Create an instance of Transmon if transmon is None, otherwise update the value of n_g.
        if not transmon:
            transmon = Transmon(n = N, e_c = e_c, e_j = e_j, n_g = n_g)
        else:
            transmon.n_g = n_g

        # Normalize the diagonalized Hamiltonian, by dividing each entry in the matrix by e_c.
        h_diagonalized_normalized = np.divide(transmon.h_diagonalized, e_c)

        # Get the eigenvalues of the Hamiltonian and store them in a row of eigenvalues matrix.
        eigenvalues_i = h_diagonalized_normalized.diagonal()
        eigenvalues[ i, : ] = eigenvalues_i
    # Sort each row of eigenvalues matrix.
    # Required as h_diagonalized, computed by np.linalg.eig, is not necessarily ordered.
    eigenvalues = np.sort(eigenvalues, axis = 1)

    return eigenvalues


def plot_hamiltonian(eigenvalues, e_j=None, e_c=None):
    for n in range(N):
        plt.plot(N_G_RANGE, eigenvalues[ :, n ], '--', lw = 1.5)

    plt.title(
        f'H eigenvals vs $n_g$. $E_J$={e_j}, $E_C$={e_c}, $N$={N}, '
        f'datapoints={int((N_G_UPPER - N_G_LOWER) / N_G_RES)}.'
    )
    plt.xlabel('$n_g$')
    plt.ylabel('Eigenvalues')
    plt.grid(True)
    plt.show()


def case_A():
    # Plot case E_J/E_C = 1
    e_j = 1.0
    e_c = 1.0
    eigenvalues = compute_eigenvalues(e_c = e_c, e_j = e_j)
    plot_hamiltonian(eigenvalues = eigenvalues, e_j = e_j, e_c = e_c)


def case_B():
    # Plot case E_J/E_C = 10
    e_j = 10.0
    e_c = 1.0
    eigenvalues = compute_eigenvalues(e_c = e_c, e_j = e_j)
    plot_hamiltonian(eigenvalues = eigenvalues, e_j = e_j, e_c = e_c)


def case_C():
    # Plot case E_J/E_C = 100
    e_j = 100.0
    e_c = 1.0
    eigenvalues = compute_eigenvalues(e_c = e_c, e_j = e_j)
    plot_hamiltonian(eigenvalues = eigenvalues, e_j = e_j, e_c = e_c)


if __name__ == '__main__':
    case_A()
    case_B()
    case_C()
