from galois import GF, FieldArray
from typing import List
from .polynomial import Polynomial
from bposd.css import css_code

import numpy as np


class BivariateBicycleCode:

    def __init__(self, l, m, poly_a: Polynomial, poly_b: Polynomial):
        self.l = l
        self.m = m
        self.n_data = l * m

        self.poly_a = poly_a
        self.poly_b = poly_b
        self.A = self._poly_to_matrix(poly_a)
        self.B = self._poly_to_matrix(poly_b)

        GF2 = GF(2)
        self.hx = GF2(np.hstack((self.A, self.B)).astype(int))
        self.hz = GF2(np.hstack((self.B.T, self.A.T)).astype(int))
        qcode = css_code(hx=self.hx, hz=self.hz)
        self.lx = qcode.lx
        self.lz = qcode.lz

        rank_hx = np.linalg.matrix_rank(self.hx)
        rank_hz = np.linalg.matrix_rank(self.hz)
        self.n = 2 * self.l * self.m
        self.k = self.n - rank_hx - rank_hz
        self.d = None

        assert self._check_commutation(
            self.hx, self.hz
        ), "The provided polynomials do not make commutative matrices, invalid CSS code."

    def _poly_to_matrix(self, poly: Polynomial):
        matrix = np.zeros((self.n_data, self.n_data), dtype=int)
        for r in range(self.n_data):
            rx, ry = divmod(r, self.m)
            for exp_x, exp_y in poly.terms:
                new_rx = (rx + exp_x) % self.l
                new_ry = (ry + exp_y) % self.m

                c = new_rx * self.m + new_ry
                matrix[r, c] ^= 1

        return matrix

    def _check_commutation(self, hx: np.ndarray, hz: np.ndarray):

        return np.all(hx @ hz.T == 0)

    @classmethod
    def generate_code(cls, distance: int):
        SUPPORTED_BB = {
            4: {
                "l": 3,
                "m": 3,
                "poly_a": "x^1 + y^0 + y^2",
                "poly_b": "y^1 + x^0 + x^2",
            },
            6: {
                "l": 6,
                "m": 6,
                "poly_a": "x^3 + y^1 + y^2",
                "poly_b": "y^3 + x^1 + x^2",
            },
            12: {
                "l": 12,
                "m": 6,
                "poly_a": "x^3 + y^1 + y^2",
                "poly_b": "y^3 + x^1 + x^2",
            },
            18: {
                "l": 12,
                "m": 12,
                "poly_a": "x^3 + y^2 + y^7",
                "poly_b": "y^3 + x^1 + x^2",
            },
        }

        if distance in SUPPORTED_BB:
            return cls(
                l=SUPPORTED_BB[distance]["l"],
                m=SUPPORTED_BB[distance]["m"],
                poly_a=Polynomial(SUPPORTED_BB[distance]["poly_a"]),
                poly_b=Polynomial(SUPPORTED_BB[distance]["poly_b"]),
            )
        else:
            raise NotImplementedError(f"Distance: {distance} is not supported.")

    def L_support(self, i, j):
        return 6*i + j
    
    def R_support(self, i, j):
        return 72 + 6*i + j

    def support(self, p: Polynomial, q: Polynomial):
        supports = []
        for term in p.terms:
            idx = self.L_support(*term)
            supports.append(idx)
        
        for term in q.terms:
            idx = self.R_support(*term)
            supports.append(idx)

        return supports 


class SurfaceCode:
    hx: FieldArray
    hz: FieldArray

    def __init__(self, distance: int):
        if (
            isinstance(distance, bool)
            or not isinstance(distance, (int, np.integer))
            or distance < 3
            or distance % 2 == 0
        ):
            raise ValueError("distance must be an odd integer >= 3")

        self.d = int(distance)
        self.n = self.d**2
        self.k = 1

        GF2 = GF(2)
        x_supports = []
        z_supports = []

        def index(row, col):
            return row * self.d + col

        # Interior weight-4 checks: alternating X and Z squares.
        for row in range(self.d - 1):
            for col in range(self.d - 1):
                support = [
                    index(row, col),
                    index(row, col + 1),
                    index(row + 1, col),
                    index(row + 1, col + 1),
                ]

                if (row + col) % 2 == 0:
                    x_supports.append(support)
                else:
                    z_supports.append(support)

        # Boundary weight-2 X checks: right/left, alternating.
        for row in range(self.d - 1):
            col = self.d - 1 if row % 2 == 0 else 0
            x_supports.append([
                index(row, col),
                index(row + 1, col),
            ])

        # Boundary weight-2 Z checks: top/bottom, alternating.
        for col in range(self.d - 1):
            row = 0 if col % 2 == 0 else self.d - 1
            z_supports.append([
                index(row, col),
                index(row, col + 1),
            ])

        self.hx = GF2.Zeros((len(x_supports), self.n))
        self.hz = GF2.Zeros((len(z_supports), self.n))

        for check, support in enumerate(x_supports):
            self.hx[check, support] = 1

        for check, support in enumerate(z_supports):
            self.hz[check, support] = 1

        assert np.all(self.hx @ self.hz.T == 0)

        assert (
            self.n
            - np.linalg.matrix_rank(self.hx)
            - np.linalg.matrix_rank(self.hz)
        ) == self.k

    def logical_qubit_support(self, pauli: str = "Z") -> list[int]:
        """
        Return a representative logical operator's local support.

        X: top row.
        Z: left column.
        """
        pauli = pauli.upper()

        if pauli == "X":
            return list(range(self.d))

        if pauli == "Z":
            return [row * self.d for row in range(self.d)]

        raise ValueError("pauli must be 'X' or 'Z'")


def nnz(array):
    nnz_indices = []
    for i, ele in enumerate(array):
        if ele != 0:
            nnz_indices.append(i)

    return nnz_indices