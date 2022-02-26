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
        sparse.kron(partial_A, sparse.identity(partial_B.shape[1])),
        sparse.kron(sparse.identity(partial_A.shape[1]), partial_B)
    ])
    
    partial_1_factors = [
        sparse.kron(sparse.identity(partial_A.shape[0]), partial_B),
        sparse.kron(partial_A, sparse.identity(partial_B.shape[0]))
    ]

    partial_1 = sparse.hstack(partial_1_factors)

    if check_complex:
        assert np.all((partial_1 @ partial_2).data % 2 == 0)

    if compute_logicals is None:
        compute_logicals = True

    x_logicals = []
    z_logicals = []
    if compute_logicals:
        # Find the logicals as Im D^A x Ker D^B + Ker D^A x Im D^B
        partial_A_dense = partial_A.todense()
        partial_B_dense = partial_B.todense()
        
        # rows of gen_A are the basis
        gen_A = get_generator_matrix(GF2(partial_A_dense))
        gen_B = get_generator_matrix(GF2(partial_B_dense))

        partial_A_dual_dense = partial_A_dense.transpose()
        partial_B_dual_dense = partial_B_dense.transpose()

        # Rows of the generator for the transpose code are a basis for the orthogonal complement of the image of the check matrix 
        gen_A_dual = get_generator_matrix(GF2(partial_A_dense.transpose()))
        gen_B_dual = get_generator_matrix(GF2(partial_B_dense.transpose()))

        # Logicals for the dual
        z_logicals.extend(np.hstack([np.kron(gen_A_dual[i, :], gen_B[j, :]), np.zeros(partial_A_dense.shape[1]*partial_B_dual_dense.shape[1])]).astype(np.uint8) for i in range(gen_A_dual.shape[0]) for j in range(gen_B.shape[0]))
        z_logicals.extend(np.hstack([np.zeros(gen_A_dual.shape[1] * partial_B_dense.shape[1]), np.kron(gen_A[i, :], gen_B_dual[j, :])]).astype(np.uint8) for i in range(gen_A.shape[0]) for j in range(gen_B_dual.shape[0]))

        # Don't worry about the non self-dual case for now
        assert (partial_A != partial_B.transpose()).nnz == 0
        x_logicals = z_logicals
        

    logicals = (x_logicals, z_logicals, len(x_logicals))

    # C2 dimension
    assert partial_2.shape[1] == partial_A.shape[1]*partial_B.shape[1]
    # C1 dimension
    assert partial_1.shape[1] == partial_A.shape[0]*partial_B.shape[1] + partial_A.shape[1]*partial_B.shape[0]
    assert partial_1.shape[1] == partial_2.shape[0]
    # C0 dimension
    assert partial_1.shape[0] == partial_A.shape[0]*partial_B.shape[0]

    assert(len(x_logicals) == len(z_logicals))

    return ((partial_2.tocsc().astype(np.uint8), partial_1.tocsr().astype(np.uint8), num_cols(partial_1)), logicals)
