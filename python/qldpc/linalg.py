from .qecc_util import GF2
import numpy as np
from typing import List, Optional

def gf2_linsolve(A : np.array, b : np.array) -> Optional[np.array]:
    '''Return x s.t. Ax = b over GF(2) if a solution exists'''

    # Convert A to RREF
    augmented = GF2(np.hstack([A, b[:,np.newaxis]]))
    aug_rref = augmented.row_reduce(ncols = min(augmented.shape[1]-1, augmented.shape[0]))
    # Use RREF to try to reduce b (Is this right?)
    # We should probably just evaluate a mat-vec product on the RREF'd subspace
    aug_rref_rank = np.sum(~np.all(aug_rref == 0, axis=1))
    solution_vec = aug_rref.tranpose().row_reduce(ncols = aug_rref_rank)[aug_rref.shape[1], :]

    if (A@solution_vec)%2 == b:
        return solution_vec
    else:
        return None

def gf2_get_pivots(A : np.array) -> List[int]:
    largest_index = (A!=0).argmax(axis=1)
    return np.extract(A[range(A.shape[0]), largest_index]!=0, largest_index)


def get_rank(A : np.array) -> int:
    return np.linalg.matrix_rank(GF2(A))
