from qldpc.qldpc import row_reduce
import numpy as np

def test_gf2_row_reduce():
    rand = np.random.default_rng(seed=412)
    for rows in [2, 4, 8, 16, 32]:
        for cols in [2, 4, 8, 16, 32]:
            for _ in range(200):
                A = np.where(rand.random((rows, cols)) < 0.3, 1, 0).astype(np.uint8)
                A = row_reduce(A)

                # Check reduced row echeleon
                last_pivot_col = None
                for i in range(rows):
                    nonzeros = np.nonzero(A[i, :])[0]

                    if nonzeros.shape[0] == 0:
                        last_pivot_col = cols-1
                    else:
                        next_pivot_col = nonzeros[0]
                        # The pivot should be the only 1 in the column
                        assert(len(np.nonzero(A[:, next_pivot_col])[0]) == 1)
                        if last_pivot_col is not None:
                            assert(next_pivot_col > last_pivot_col)
                        last_pivot_col = next_pivot_col
