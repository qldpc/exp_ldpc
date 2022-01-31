import scipy.sparse as sparse
import networkx as nx
import itertools
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows, GF2
from .generator_matrix import get_generator_matrix

def homological_product(partial_A : sparse.spmatrix, partial_B : sparse.spmatrix, check_complex = None, compute_logicals = None) -> (QuantumCodeChecks, QuantumCodeLogicals):
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

    if compute_logicals is None:
        compute_logicals = True

    x_logicals = []
    z_logicals = []
    if compute_logicals:
        # rows of gen_A are the basis
        gen_A = get_generator_matrix(GF2(partial_A.todense()))
        gen_B = get_generator_matrix(GF2(partial_B.todense()))

        x_logicals.extend(np.kron(gen_A[i, :], gen_B[j, :]) for i in range(gen_A.shape[0]) for j in range(gen_B.shape[0]))

        gen_A_dual = get_generator_matrix(GF2(partial_A.todense().transpose()))
        gen_B_dual = get_generator_matrix(GF2(partial_B.todense().transpose()))
        
        z_logicals.extend(np.kron(gen_A_dual[i, :], gen_B_dual[j, :]) for i in range(gen_A_dual.shape[0]) for j in range(gen_B_dual.shape[0]))

    logicals = (x_logicals, z_logicals, len(x_logicals))

    return ((partial_2.tocsc(), partial_1.tocsr(), num_cols(partial_1)), logicals)
