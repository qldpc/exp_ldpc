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

        # Zero vectors restricted to the A_1 x B_0 and A_0 x B_1 subspaces
        A1B0_zero = np.zeros((1, partial_A_dense.shape[1] * partial_B_dense.shape[0]))
        A0B1_zero = np.zeros((1, partial_A_dense.shape[0] * partial_B_dense.shape[1]))

        A1B0_logicals = [np.hstack([A0B1_zero, np.kron(gen_A[i, :], partial_B_dense[j, :])]) for i in range(gen_A.shape[0]) for j in range(partial_B_dense.shape[0])]
        A0B1_logicals = [np.hstack([np.kron(partial_A_dense[i, :], gen_B[j, :]), A1B0_zero]) for i in range(partial_A_dense.shape[0]) for j in range(gen_B.shape[0])]

        x_logicals.extend(A1B0_logicals)
        x_logicals.extend(A0B1_logicals)

        # Logicals for the dual

        partial_A_dual_dense = partial_A_dense.transpose()
        partial_B_dual_dense = partial_B_dense.transpose()

        gen_A_dual = get_generator_matrix(GF2(partial_A_dense.transpose()))
        gen_B_dual = get_generator_matrix(GF2(partial_B_dense.transpose()))

        A1B0_dual_zero = np.zeros((1, partial_A_dual_dense.shape[1] * partial_B_dual_dense.shape[0]))
        A0B1_dual_zero = np.zeros((1, partial_A_dual_dense.shape[0] * partial_B_dual_dense.shape[1]))

        A1B0_dual_logicals = [np.hstack([A1B0_dual_zero, np.kron(gen_A_dual[i, :], partial_B_dual_dense[j, :])]) for i in range(gen_A_dual.shape[0]) for j in range(partial_B_dual_dense.shape[0])]
        A0B1_dual_logicals = [np.hstack([np.kron(partial_A_dual_dense[i, :], gen_B_dual[j, :]), A1B0_dual_zero]) for i in range(partial_A_dual_dense.shape[0]) for j in range(gen_B_dual.shape[0])]
        
        z_logicals.extend(A1B0_dual_logicals)
        z_logicals.extend(A0B1_dual_logicals)

    logicals = (x_logicals, z_logicals, len(x_logicals))

    # C2 dimension
    assert partial_2.shape[1] == partial_A.shape[1]*partial_B.shape[1]
    # C1 dimension
    assert partial_1.shape[1] == partial_A.shape[0]*partial_B.shape[1] + partial_A.shape[1]*partial_B.shape[0]
    assert partial_1.shape[1] == partial_2.shape[0]
    # C0 dimension
    assert partial_1.shape[0] == partial_A.shape[0]*partial_B.shape[0]

    return ((partial_2.tocsc(), partial_1.tocsr(), num_cols(partial_1)), logicals)
