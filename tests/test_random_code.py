from qldpc.random_code import random_check_matrix

def test_random_check_matrix():
    r = 20
    n = 40
    H = random_check_matrix(r, n, seed=42)
    assert H.shape == (r,n)
