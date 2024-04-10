from typing import Tuple

import numpy as np
from numpy import ndarray


class Transmon:
    """The Transmon class.

    Transmon class implements the charge-basis matrix representation of the transmon.
    Transmon supports returning the Hamiltonian given parameters: n, e_c, e_j & n_g.
    Additionally, the class supports returning a diagonalized form of the Hamiltonian.

    The Hamiltonian is defined as:
    H = lim_{N->inf} Σ_{n=-N}^{n=N} [ 4*E_C*(n − n_g)^2 |i⟩⟨i| − E_J/2 (|i+1⟩⟨i| + H.c) ]
    E_C, E_J and n_g are parameters of the matrix.
    + H.c term stands for plus the Hermitian conjugate.
    """

    def __init__(self, n: int, e_c: float, e_j: float, n_g: float) -> None:
        """
        Args:
            n: the upper summation bound.
            e_c: E_C Hamiltonian parameter.
            e_j: E_J Hamiltonian parameter.
            n_g: n_g Hamiltonian parameter.

        Raises:
            TypeError: If the summation bound, n, is not an integer.
            ValueError: If the summation bound has a negative value.
            In this case, the size of the Hamiltonian of 2n+1 by 2n+1 and the summation is invalid.
        """
        if not isinstance(n, int):
            raise TypeError("The value of n should have type int")

        if n < 0:
            raise ValueError("The value of n should be non-negative")

        self.__n = n
        self.__e_c = e_c
        self.__e_j = e_j
        self.__n_g = n_g

        # Initialize instance variables responsible for the Hamiltonian
        self.__h = None
        self.__h_diagonalized = None
        self.__h_eigenbasis = None

        # Compute and set values for the variables responsible for the Hamiltonian
        self._compute_and_set_hamiltonian()

    def _compute_and_set_hamiltonian(self) -> None:
        """Compute the Hamiltonian and diagonalized Hamiltonian and set respective instance
        variables."""
        self.__h = self._compute_hamiltonian()
        self.__h_diagonalized, self.__h_eigenbasis = self._diagonalize_hamiltonian(self.__h)

    def _compute_hamiltonian(self) -> ndarray:
        """Compute the Hamiltonian in matrix representation.
        Returns:
            The computed Hamiltonian as 2n+1 by 2n+1 ndarray.
        """
        # Initialize a 2n+1 by 2n+1 matrix filled with zeros
        hamiltonian = np.zeros((2 * self.__n + 1, 2 * self.__n + 1), dtype = np.float64)

        # (n,n)-th entry of the Hamiltonian is 4*E_C*(n − n_g)^2, as given by the formula
        diagonal_entries = np.array(
            [ 4 * self.__e_c * np.power(n - self.__n_g, 2) for n in
              np.arange(start = -self.__n, stop = self.__n + 1, step = 1, dtype = int) ]
        )

        # (n, n-1)-th & (n, n+1)-th entries of the Hamiltonian are -E_J/2, as given by the formula
        upper_and_lower_diagonal_entries = -self.__e_j / 2

        # Fill the main diagonal of the Hamiltonian
        np.fill_diagonal(hamiltonian, diagonal_entries)

        # Fill the upper diagonal of the Hamiltonian
        np.fill_diagonal(hamiltonian[ :-1, 1: ], upper_and_lower_diagonal_entries)

        # Fill the lower diagonal of the Hamiltonian
        np.fill_diagonal(hamiltonian[ 1:, :-1 ], upper_and_lower_diagonal_entries)

        # The remaining entries are zero, hence the computation is complete!
        return hamiltonian

    @staticmethod
    def _diagonalize_hamiltonian(hamiltonian) -> Tuple[ ndarray, ndarray ]:
        """Compute the eigenvalues and eigenbasis of the Hamiltonian.
        Args:
            hamiltonian: the computed Hamiltonian represented as 2n+1 by 2n+1 ndarray.

        Returns:
            The 2n+1 by 2n+1 diagonal matrix with eigenvalues of the Hamiltonian as entries,
            and the corresponding eigenbasis as 2n+1 by 2n+1 ndarray.
        """
        h_eigenvalues, h_eigenbasis = np.linalg.eig(hamiltonian)
        return np.diag(h_eigenvalues), h_eigenbasis

    def __repr__(self) -> str:
        """Return the string representation of the Transmon class."""
        return f"N={self.__n}, E_C={self.__e_c}, E_J={self.__e_j}, n_g={self.__n_g}." \
               f"\n Hamiltonian: \n {self.__h} " \
               f"\n Eigenvalues: \n {self.__h_diagonalized.diagonal()}" \
               f"\n Eigenbasis: \n {self.__h_eigenbasis}"

    @property
    def h(self) -> ndarray:
        """The computed Hamiltonian for the current values of n, e_c, e_j & n_g.

        Returns:
            The Hamiltonian as 2n+1 by 2n+1 matrix, with i,j-th entry representing the coefficient
            in front of |i><j|.
        """
        return self.__h

    @property
    def h_diagonalized(self) -> ndarray:
        """The diagonalized Hamiltonian in its eigenbasis.

        Returns:
            The diagonal matrix with eigenvalues of the Hamiltonian as diagonal entries.
        """
        return self.__h_diagonalized

    @property
    def h_eigenbasis(self) -> ndarray:
        """The eigenbasis of the Hamiltonian.

        Returns:
            The 2n+1 by 2n+1 matrix with eigenvectors to the Hamiltonian as columns.

        Note:
            The order of the eigenvectors corresponds to the order of eigenvalues in
            the diagonalized Hamiltonian.
        """
        return self.__h_eigenbasis

    @property
    def n_g(self) -> float:
        """n_g parameter of the current instance of Transmon.

        Returns:
            The value of n_g.
        """
        return self.__n_g

    @n_g.setter
    def n_g(self, n_g: float) -> None:
        """Set n_g parameter of the current instance of Transmon.

        Args:
            n_g: the value of n_g parameter of type float.

        Note:
            The Hamiltonian has to be recomputed, as one of the matrix parameters
            is being modified.
        """
        self.__n_g = n_g
        self._compute_and_set_hamiltonian()

    @property
    def e_j(self) -> float:
        """e_j parameter of the current instance of Transmon.

        Returns:
            The value of e_j.
        """
        return self.__e_j

    @e_j.setter
    def e_j(self, e_j: float) -> None:
        """Set e_j parameter of the current instance of Transmon.

        Args:
            e_j: the value of e_c parameter of type float.

        Note:
            The Hamiltonian has to be recomputed, as one of the matrix parameters
            is being modified.
        """
        self.__e_j = e_j
        self._compute_and_set_hamiltonian()

    @property
    def e_c(self) -> float:
        """e_c parameter of the current instance of Transmon.

        Returns:
            The value of e_c.
        """
        return self.__e_c

    @e_c.setter
    def e_c(self, e_c: float) -> None:
        """Set e_c parameter of the current instance of Transmon.

        Args:
            e_c: the value of e_c parameter of type float.

        Note:
            The Hamiltonian has to be recomputed, as one of the matrix parameters
            is being modified.
        """
        self.__e_c = e_c
        self._compute_and_set_hamiltonian()

    @property
    def n(self) -> int:
        """The lower and upper summation bounds during the calculation of the Hamiltonian.

        Returns:
            The lower and upper summation bound.

        Note:
            The parameter also defines the dimensions of the Hamiltonian matrix (2n+1 by 2n+1).
        """
        return self.__n

    @n.setter
    def n(self, n: int) -> None:
        """Set the lower and upper summation bounds during the calculation of the Hamiltonian.

        Args:
            n: the upper summation bound.

        Raises:
            TypeError: If the summation bound, n, is not an integer.
            ValueError: If the summation bound has a negative value.
            In this case, the size of the Hamiltonian of 2n+1 by 2n+1 and the summation is invalid.

        Note:
            The Hamiltonian has to be recomputed, as one of the matrix parameters
            is being modified.
        """
        if not isinstance(n, int):
            raise TypeError("The value of n should have type int")

        if n < 0:
            raise ValueError("The value of n should be non-negative")

        self.__n = n
        self._compute_and_set_hamiltonian()
