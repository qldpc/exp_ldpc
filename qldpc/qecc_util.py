from typing import Iterable, Tuple
from scipy import sparse
import numpy as np
from galois import GF
from dataclasses import dataclass

GF2 = GF(2)

@dataclass
class QuantumCodeChecks:
    x : sparse.spmatrix
    z : sparse.spmatrix
    num_qubits : int

@dataclass
class QuantumCodeLogicals:
    x : np.array
    z : np.array
    num_qubits : int

def num_rows(a : np.array) -> int:
    assert(len(a.shape) == 2)
    return a.shape[0]

def num_cols(a : np.array) -> int:
    assert(len(a.shape) == 2)
    return a.shape[1]

def make_check_matrix(checks : Iterable[Iterable[int]], num_qubits) -> sparse.csr_matrix:
    '''Given check matrix specified as non-zero entries, construct a scipy sparse matrix'''
    coo_entries = np.array([[row_index, v, 1] for (row_index, row) in enumerate(checks) for v in row])
    return sparse.csr_matrix((coo_entries[:,2], (coo_entries[:,0], coo_entries[:,1])), shape=(len(checks), num_qubits), dtype=np.int32)
