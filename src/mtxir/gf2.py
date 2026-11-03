"""GF(2) helpers shared by the code and IR modules."""

import numpy as np
from galois import GF, FieldArray

GF2 = GF(2)


def block_diag_gf2(matrices):
    """Block-diagonal GF(2) matrix of the given matrices, in order."""
    matrices = list(matrices)

    total_rows = sum(A.shape[0] for A in matrices)
    total_cols = sum(A.shape[1] for A in matrices)

    H = GF2.Zeros((total_rows, total_cols))

    row = col = 0
    for A in matrices:
        rows, cols = A.shape
        H[row:row + rows, col:col + cols] = A
        row += rows
        col += cols

    return H


def check_commute(hx: FieldArray, hz: FieldArray):
    """Assert that every X check commutes with every Z check."""
    assert hx.shape[1] == hz.shape[1], "Qubit counts must match."
    assert np.all(hx @ hz.T == 0), "X and Z checks anticommute."
