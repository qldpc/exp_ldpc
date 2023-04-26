from qldpc.random_code import random_check_matrix
import numpy as np

def test_random_check_matrix():
    r = 20
    n = 40
    H = random_check_matrix(r, n, seed=42)
    assert H.shape == (r,n)

def test_random_check_matrix_full_rank():
    r = 2
    n = 5
    for i in range(1000):
        H = random_check_matrix(r, n, seed=i, full_rank=True)
        assert np.linalg.matrix_rank(H) == min(r,n)

