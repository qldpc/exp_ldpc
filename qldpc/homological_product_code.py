import scipy.sparse as sparse
import networkx as nx
import itertools
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows, GF2
from .generator_matrix import get_generator_matrix, gf2_coker, gf2_homology_generators

def compute_homology_reps(partial_A : GF2, partial_B : GF2) -> GF2:
    '''Compute representatives of the homology group of the total complex of A and B'''

    # Each row is a basis vector for the kernel / cokernel
    a_kernel = partial_A.null_space()
    b_kernel = partial_B.null_space()

    a_cokernel = gf2_coker(partial_A)
    b_cokernel = gf2_coker(partial_B)

    A0B1_zero = np.zeros(partial_A.shape[0] * partial_B.shape[1], dtype=np.uint8)
    A1B0_zero = np.zeros(partial_A.shape[1] * partial_B.shape[0], dtype=np.uint8)

    # Since the factors are 2-complexes, H_1 is (coker x ker) + (ker x coker)
    hom_reps = []
    hom_reps.extend(np.hstack([A0B1_zero, np.kron(a_kernel[i,:], b_cokernel[j,:])]) for i in range(a_kernel.shape[0]) for j in range(b_cokernel.shape[0]))
    hom_reps.extend(np.hstack([np.kron(a_cokernel[i,:], b_kernel[j,:]), A1B0_zero]) for i in range(a_cokernel.shape[0]) for j in range(b_kernel.shape[0]))

    return hom_reps

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
        compute_logicals = False

    x_logicals = []
    z_logicals = []
    if compute_logicals:
        # This codepath does not pass all tests
        raise NotImplementedError

        # Find the logicals as Im D^A x Ker D^B + Ker D^A x Im D^B
        partial_A_gf2 = GF2(partial_A.todense())
        partial_B_gf2 = GF2(partial_B.todense())

        x_logicals = compute_homology_reps(partial_A_gf2, partial_B_gf2)
        z_logicals = compute_homology_reps(partial_B_gf2.transpose(), partial_A_gf2.transpose())

        # x_logicals = gf2_homology_generators(partial_1.todense(), partial_2.todense())
        # z_logicals = gf2_homology_generators(partial_2.todense().transpose(), partial_1.todense().transpose())
    logicals = (x_logicals, z_logicals, len(x_logicals))

    # C2 dimension
    assert partial_2.shape[1] == partial_A.shape[1]*partial_B.shape[1]
    # C1 dimension
    assert partial_1.shape[1] == partial_A.shape[0]*partial_B.shape[1] + partial_A.shape[1]*partial_B.shape[0]
    assert partial_1.shape[1] == partial_2.shape[0]
    # C0 dimension
    assert partial_1.shape[0] == partial_A.shape[0]*partial_B.shape[0]

    assert len(x_logicals) == len(z_logicals)

    # Check number of logicals + number of checks == number of qubits
    if compute_logicals:
        assert len(x_logicals) + partial_2.shape[1] + partial_1.shape[0] == partial_2.shape[0]

    return ((partial_2.tocsc().astype(np.uint8), partial_1.tocsr().astype(np.uint8), num_cols(partial_1)), logicals)
