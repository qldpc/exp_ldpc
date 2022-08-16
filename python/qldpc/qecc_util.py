from typing import Iterable, Tuple
from scipy import sparse
import numpy as np
from galois import GF
from dataclasses import dataclass, field
from warnings import warn

GF2 = GF(2)

def _check_integral(matrix):
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError('Got numpy object with non-integral dtype')
    
    if np.issubdtype(matrix.dtype, np.signedinteger):
        warn('Got numpy object with signed integer datatype. This could cause problems due when overflowing')

@dataclass(frozen=True)
class QuantumCodeChecks:
    x : sparse.csr_matrix
    z : sparse.csr_matrix

    @property
    def num_qubits(self) -> int:
        '''Number of qubits the checks are defined with respect to'''
        return self.x.shape[1]

    def __init__(self, x : sparse.spmatrix, z : sparse.spmatrix):
        # Optimize format + make readonly
        x = x.tocsr()
        x.sort_indices()
        x.sum_duplicates()
        x.prune()
        x.data.flags.writeable = False
        object.__setattr__(self, 'x', x)

        z = z.tocsr()
        z.sort_indices()
        z.sum_duplicates()
        z.prune()
        z.data.flags.writeable = False
        object.__setattr__(self, 'z', z)

        # Check for integral datatypes (warn if signed)
        _check_integral(self.x)
        _check_integral(self.z)

        # Size checks
        if self.x.shape[1] != self.z.shape[1]:
            raise ValueError("x and z checks act on an inconsistent number of qubits")

@dataclass(frozen=True)
class QuantumCodeLogicals:
    x : np.array
    z : np.array

    def __post_init__(self):
        # Make arrays read only
        self.x.flags.writeable = False
        self.z.flags.writeable = False

        # Check for integral datatypes (warn if signed)
        _check_integral(self.x)
        _check_integral(self.z)

        # Size checks
        if self.x.shape[1] != self.z.shape[1]:
            raise ValueError("x and z logicals act on an inconsistent number of qubits")
        if self.x.shape[0] != self.z.shape[0]:
            raise ValueError("Number of provided X and Z logical operators mismatch")

    @property
    def num_qubits(self) -> int:
        '''Number of qubits the logicals are defined with respect to'''
        return self.x.shape[1]

    @property
    def num_logicals(self) -> int:
        '''Number of logical qubits'''
        return self.x.shape[0]

    @classmethod
    def empty(num_qubits : int):
        '''Creates a QuantumCodeLogicals object with no logical operators'''
        return QuantumCodeLogicals(
                np.zeros((0,num_qubits),dtype=np.int32),
                np.zeros((0,num_qubits),dtype=np.int32))
                

@dataclass(frozen=True)
class QuantumCode:
    checks : QuantumCodeChecks
    logicals : QuantumCodeLogicals
    
    @property
    def num_qubits(self) -> int:
        '''Number of physical qubits the code is defined with respect to'''
        return self.checks.num_qubits

    @property
    def num_logicals(self) -> int:
        '''Number of logical qubits the code encodes'''
        return self.logicals.num_logicals

    def __init__(self, checks : QuantumCodeChecks, logicals : QuantumCodeLogicals = None):
        if logicals is None:
            # Create empty set of logicals
            logicals = QuantumCodeLogicals.empty(checks.num_qubits)

        if checks.num_qubits != logicals.num_qubits:
            raise ValueError("Number of qubits for checks and logicals is inconsistent")

        object.__setattr__(self, 'checks', checks)
        object.__setattr__(self, 'logicals', logicals)

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
