import math
import unittest

import numpy as np
from ddt import data, unpack, ddt
from numpy import ndarray

from Transmon import Transmon


@ddt
class TestHamiltonianComputation(unittest.TestCase):
    """Hamiltonian computation tests."""

    @data(
        (1, 0.0, 0.0, 0.0),
        (1, 1.0, 1.0, 1.0),
        (2, 1.5, 7.0, 15.0),
        (3, 120.5, 15.3, -2.5),
        (15, 5.2, -8.9, 45.1),
        (1000, -154.0, 12.5, 3.2)
    )
    @unpack
    def test_compute_hamiltonian(self, n, e_c, e_j, n_g):
        """Test the Hamiltonian computed has the right data type, shape and correct values of the entries."""
        transmon = Transmon(n = n, e_c = e_c, e_j = e_j, n_g = n_g)

        assert isinstance(transmon.h, ndarray)
        assert transmon.h.shape == (2 * n + 1, 2 * n + 1)
        for i in range(2 * n + 1):
            for j in range(2 * n + 1):
                if i == j:
                    assert np.equal(transmon.h[ i, j ],
                                    4 * e_c * ((i - n - n_g) ** 2))
                elif j == i + 1 or j == i - 1:
                    assert np.equal(transmon.h[ i, j ],
                                    -1 * e_j / 2)
                else:
                    assert np.equal(transmon.h[ i, j ],
                                    0)

    @data(
        (0, 0.0, 0.0, 0.0, [ 0 ]),
        (1, 1.0, 0.0, 1.0, [ 4 * 2 ** 2, 0, 4 ]),
        (2, 2.0, 0.0, 3.0,
         [ 4 * 2 * (-2 - 3.0) ** 2, 4 * 2 * (-1 - 3.0) ** 2, 4 * 2 * (0 - 3.0) ** 2, 4 * 2 * (1 - 3.0) ** 2,
           4 * 2 * (2 - 3.0) ** 2 ]),
        # The matrix was computed "manually", using the formula and eigenvalues obtained using an external tool
        (3, 120.0, 15.0, -2.5, [ 127.471, 112.471, 1080.029, 3000.010, 5880.005, 9720.003, 14520.012 ]),
    )
    @unpack
    def test_diagonalize_hamiltonian(self, n, e_c, e_j, n_g, expected):
        """Test the diagonalized Hamiltonian has the correct eigenvalues."""
        transmon = Transmon(n = n, e_c = e_c, e_j = e_j, n_g = n_g)
        assert (
                np.sort(np.around(transmon.h_diagonalized.diagonal(), decimals = 3)) ==
                np.sort(np.array(expected))
        ).all()

    @data(
        (-1, ValueError), (-math.inf, TypeError), (3.4, TypeError), ("3", TypeError)
    )
    @unpack
    def test_throws_exception(self, n, exception_type):
        """Test the Transmon class does not allow invalid n."""
        with self.assertRaises(exception_type):
            transmon = Transmon(n = n, e_c = 0, e_j = 0, n_g = 0)

    @data(
        ("e_c", 1.5), ("e_j", 3.8), ("n_g", 2.0), ("n", 15)
    )
    @unpack
    def test_update_instance_variable(self, variable_to_update, value):
        """Test the Transmon class handles updating instance variables correctly and recomputes the Hamiltonian."""
        n = 2
        e_j = 1.0
        e_c = 1.0
        n_g = 1.0
        transmon = Transmon(n = n, e_c = e_c, e_j = e_j, n_g = n_g)

        if variable_to_update == "e_c":
            transmon.e_c = value
            new_transmon = Transmon(n = n, e_c = value, e_j = e_j, n_g = n_g)
        elif variable_to_update == "e_j":
            transmon.e_j = value
            new_transmon = Transmon(n = n, e_c = e_c, e_j = value, n_g = n_g)
        elif variable_to_update == "n_g":
            transmon.n_g = value
            new_transmon = Transmon(n = n, e_c = e_c, e_j = e_j, n_g = value)
        elif variable_to_update == "n":
            transmon.n = value
            new_transmon = Transmon(n = value, e_c = e_c, e_j = e_j, n_g = n_g)

        assert repr(transmon) == repr(new_transmon)
