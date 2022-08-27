from .qecc_util import GF2
import numpy as np
from typing import List

def gf2_null_space(A : np.array) -> np.array:
    return GF2(A).null_space()

def gf2_column_space(A : np.array) -> np.array:
    return GF2(A).column_space()

def gf2_row_reduce(A : np.array, ncols : int = None) -> np.array:
    return GF2(A).row_reduce(ncols=ncols)

def gf2_get_pivots(A : np.array) -> List[int]:
    largest_index = (A!=0).argmax(axis=1)
    return np.extract(A[range(A.shape[0]), largest_index]!=0, largest_index)

def gf2_matrix_rank(A : np.array) -> int:
    return get_rank(A)

def get_rank(A : np.array) -> int:
    return np.linalg.matrix_rank(GF2(A))

