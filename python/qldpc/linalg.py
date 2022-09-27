from .qecc_util import GF2
import numpy as np
from typing import List, Optional

def gf2_linsolve(A : np.array, b : np.array) -> Optional[np.array]:
    '''Return x s.t. Ax = b over GF(2) if a solution exists'''

    # Convert A to RREF
    augmented = GF2(np.hstack([A, b[:,np.newaxis]]))
    ident_block_size = min(A.shape[0], A.shape[1])
    aug_rref = augmented.row_reduce(ncols = A.shape[1])
    pivots = gf2_get_pivots(aug_rref[:,:-1])

    candidate_soln = np.zeros(A.shape[1], dtype=np.uint32)

    candidate_soln[pivots] = np.array(aug_rref[:len(pivots), -1])

    if ((A@candidate_soln)%2 == b).all():
        return candidate_soln
    else:
        return None

def gf2_get_pivots(A : np.array) -> List[int]:
    largest_index = (A!=0).argmax(axis=1)
    return np.extract(A[range(A.shape[0]), largest_index]!=0, largest_index)

def get_rank(A : np.array) -> int:
    return np.linalg.matrix_rank(GF2(A))
