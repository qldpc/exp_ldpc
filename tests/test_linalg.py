from qldpc.linalg import gf2_row_reduce
import numpy as np


def _check_row_reduced(A):
    # Check reduced row echeleon
    last_pivot_col = None
    for i in range(A.shape[0]):
        nonzeros = np.nonzero(A[i, :])[0]

        if nonzeros.shape[0] == 0:
            last_pivot_col = A.shape[1]-1
        else:
            next_pivot_col = nonzeros[0]
            # The pivot should be the only 1 in the column
            assert(len(np.nonzero(A[:, next_pivot_col])[0]) == 1)
            if last_pivot_col is not None:
                assert(next_pivot_col > last_pivot_col)
            last_pivot_col = next_pivot_col

def test_gf2_row_reduce():
    rand = np.random.default_rng(seed=412)
    for rows in [2, 4, 8, 16, 17, 32]:
        for cols in [2, 4, 8, 16, 17, 32]:
            for _ in range(200):
                A = np.where(rand.random((rows, cols)) < 0.3, 1, 0).astype(np.uint8)
                A = gf2_row_reduce(A)
                _check_row_reduced(A)

def test_gf2_row_reduce_big():
    rows = 10000
    cols = 10000
    rand = np.random.default_rng(seed=412)
    A = np.where(rand.random((rows, cols)) < 0.5, 1, 0).astype(np.uint8)
    A = gf2_row_reduce(A, use_sage = True)
    _check_row_reduced(A)