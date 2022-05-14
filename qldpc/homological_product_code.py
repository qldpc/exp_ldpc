import scipy.sparse as sparse
import networkx as nx
import itertools
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows, GF2
from .generator_matrix import get_generator_matrix, gf2_coker, gf2_homology_generators

def compute_homology_reps(partial_A : GF2, partial_B : GF2, dual=False) -> GF2:
    '''Compute representatives of the homology group of the total complex of A and B'''

    # Each row is a basis vector for the kernel / cokernel
    a_kernel = partial_A.null_space()
    b_kernel = partial_B.null_space()

    a_cokernel = gf2_coker(partial_A)
    b_cokernel = gf2_coker(partial_B)

    A0B1_zero = np.zeros(partial_A.shape[0] * partial_B.shape[1], dtype=np.uint8)
    A1B0_zero = np.zeros(partial_A.shape[1] * partial_B.shape[0], dtype=np.uint8)

    # Since the factors are 2-complexes, H_1 is (coker x ker) + (ker x coker)
    # When we take the dual (transpose) the order of the terms in the direct sum get swapped
    sum_order = -1 if dual else 1
    
    hom_reps = []
    hom_reps.extend(np.hstack([A0B1_zero, np.kron(a_kernel[i,:], b_cokernel[j,:])][::sum_order]) for i in range(a_kernel.shape[0]) for j in range(b_cokernel.shape[0]))
    hom_reps.extend(np.hstack([np.kron(a_cokernel[i,:], b_kernel[j,:]), A1B0_zero][::sum_order]) for i in range(a_cokernel.shape[0]) for j in range(b_kernel.shape[0]))
    
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
        # Find the logicals as Im D^A x Ker D^B + Ker D^A x Im D^B
        partial_A_gf2 = GF2(partial_A.todense())
        partial_B_gf2 = GF2(partial_B.todense())

        x_logicals = compute_homology_reps(partial_A_gf2, partial_B_gf2)
        z_logicals_unpaired = np.vstack(compute_homology_reps(partial_A_gf2.transpose(), partial_B_gf2.transpose(), dual=True))

        # compute new Z logicals (by multiplying different Z logicals together as operators) s.t. each logical anticommutes with exactly one X logical
        # We do this by row reducing the matrix of inner products to the identity, keeping track of the operations
        # We form the augmented matrix (LzLx^T | Lz) (where rows are the logicals as F2 vectors)
        
        inner_products = z_logicals_unpaired @ np.vstack(x_logicals).T
        z_logicals_aug = GF2(np.hstack([inner_products, z_logicals_unpaired]))

        num_pairs = inner_products.shape[1]
        z_logicals_paired = z_logicals_aug.row_reduce(ncols=num_pairs)[:,num_pairs:]
        z_logicals = [z_logicals_paired[i,:] for i in range(z_logicals_paired.shape[0])]
       
        if check_complex:
            for l in x_logicals:
                assert np.all((partial_1 @ l) % 2 == 0)
                
            for l in z_logicals:
                assert np.all((partial_2.T @ l) % 2 == 0)
            
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
