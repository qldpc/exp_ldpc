import scipy.sparse as sparse
import networkx as nx
import itertools
import numpy as np
from .qecc_util import QuantumCodeChecks, num_cols, num_rows

def homological_product(partial_A : sparse.spmatrix, partial_B : sparse.spmatrix, check_complex = None) -> QuantumCodeChecks:
    '''Compute the homological product of two 2-complexes defined by their non-trivial boundary map
        Returns pair of boundary maps (partial_2, partial_1) of the total complex
    '''
    if check_complex is None:
        check_complex = False

    # D^A x I + I x D^B A_1 x B_1 -> A_0 x B_1 + A_1 x B_0
    partial_2 = sparse.vstack([
        sparse.kron(partial_A, sparse.identity(num_cols(partial_B))),
        sparse.kron(sparse.identity(num_cols(partial_A)), partial_B)
    ])
    
    partial_1 = sparse.hstack([
        sparse.kron(sparse.identity(num_rows(partial_A)), partial_B),
        sparse.kron(partial_A, sparse.identity(num_rows(partial_B)))
    ])

    if check_complex:
        assert np.all((partial_1 @ partial_2).data % 2 == 0)

    return (partial_2.tocsc(), partial_1.tocsr(), num_cols(partial_1))
