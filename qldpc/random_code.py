import numpy as np
from .qecc_util import GF2

def random_check_matrix(r, n, seed = None):
    '''Generate a random r x n check matrix over GF2'''
    rng = np.random.default_rng(seed)
    return GF2(rng.integers(low=0,high=2, size=(r,n)))
    
