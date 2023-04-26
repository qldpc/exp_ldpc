import numpy as np
from .qecc_util import GF2

def random_check_matrix(r, n, seed = None, full_rank = None):
    '''Generate a random r x n check matrix over GF2. If full_rank is set then rejectiong sample until the matrix is full rank'''
    if full_rank is None:
        full_rank = False
    
    rng = np.random.default_rng(seed)

    
    for _ in range(10000):
        h = GF2(rng.integers(low=0,high=2, size=(r,n)))
        
        if full_rank is True:
            if np.linalg.matrix_rank(h) == min(h.shape):
                break
        else:
            break
    else:
        raise RuntimeError('Failed to construct random matrix: Number of retries exceeded')
    return h
    
