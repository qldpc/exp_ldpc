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

    # D^A x I + I x D^B : A_1 x B_1 -> A_0 x B_1 + A_1 x B_0
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
        partial_A_gf2 = GF2(partial_A.todense())
        partial_B_gf2 = GF2(partial_B.todense())

        # Each row is a basis vector for the kernel / cokernel
        a_kernel = partial_A_gf2.null_space()
        b_kernel = partial_B_gf2.null_space()

        a_cokernel = partial_A_gf2.left_null_space()
        b_cokernel = partial_B_gf2.left_null_space()

        A0B1_zero = np.zeros(partial_A.shape[0] * partial_B.shape[1], dtype=np.uint8)
        A1B0_zero = np.zeros(partial_A.shape[1] * partial_B.shape[0], dtype=np.uint8)

        print(f'{a_kernel.shape=} {b_kernel.shape=} {a_cokernel.shape=} {b_cokernel.shape=} {A0B1_zero.shape=} {A1B0_zero.shape=}')
        
        x_logicals.extend(np.hstack([A0B1_zero, np.kron(a_kernel[i,:], b_cokernel[j,:])]) for i in range(a_kernel.shape[0]) for j in range(b_cokernel.shape[0]))
        x_logicals.extend(np.hstack([np.kron(a_cokernel[i,:], b_kernel[j,:]), A1B0_zero]) for i in range(a_cokernel.shape[0]) for j in range(b_kernel.shape[0]))

        # Don't worry about the non self-dual case for now
        assert (partial_A != partial_B.transpose()).nnz == 0
        z_logicals = x_logicals
        

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
