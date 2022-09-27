from qldpc.linalg import gf2_linsolve, gf2_get_pivots, get_rank
from qldpc.random_code import random_check_matrix
from qldpc import GF2
import numpy as np


def test_gf2_linsolve_random():
    rand = np.random.default_rng(seed=42)
    for rows in [8,16,32,64]:
        for cols in [8,16,32,64]:
            for i in range(10):
                A = np.array(random_check_matrix(rows, cols, seed=42+i))
                x = np.where(rand.random(cols) < 0.5, 1, 0).astype(np.uint32)
                b = (A @ x)%2
                output = gf2_linsolve(A, b)
                assert output is not None
                # There may be more than one solution
                assert ((A @ output)%2 == b).all()

def test_gf2_get_pivots():
    rand = np.random.default_rng(seed=42)
    for rows in [8,16,32,64]:
        for cols in [8,16,32,64]:
            for i in range(10):
                A = np.array(random_check_matrix(rows, cols, seed=42+i))
                A_rref = np.array(GF2(A).row_reduce())

                num_pivots = len(gf2_get_pivots(A_rref))
                assert num_pivots == get_rank(A_rref)
