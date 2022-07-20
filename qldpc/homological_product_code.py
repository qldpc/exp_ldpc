import scipy.sparse as sparse
import networkx as nx
import itertools
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows, GF2
from .linalg import get_rank, gf2_get_pivots

def compute_homology_reps(partial_2 : GF2, partial_1 : GF2, dual=False) -> GF2:
    '''Compute representatives of the homology group of the complex defined by partial_1 . partial_2 = 0'''

    kernel = partial_1.null_space()
    image = partial_2.column_space() # This is already row reduced

    # We have a basis for the image and we want to extend this to a basis of the kernel
    # Axler 2.B shows how to do this by extending the basis one at a time

    reduced_aug_matrix = np.hstack([image.T, kernel.T]).row_reduce()
    # The pivot columns tell us a spanning set for the vector space
    # image.T is already row reduced so the remainder must be the complement in the kernel
    pivot_cols = gf2_get_pivots(reduced_aug_matrix)
    generator_indices = pivot_cols[image.shape[0]:] - image.shape[0]

    return kernel[generator_indices,:]

def compute_logical_pairs(z_logicals : GF2, x_logicals : GF2) -> GF2:
    '''Given a set of logicals (z_logicals, x_logicals) return a new set of Z logicals so that we have pairs (Z_k, X_k) and {Z_k,X_k} = delta_kk'''

    # We do this by row reducing the matrix of inner products to the identity, keeping track of the operations
    # We form the augmented matrix (LzLx^T | Lz) (where rows are the logicals as F2 vectors)        
    inner_products = z_logicals @ x_logicals.T
    num_pairs = inner_products.shape[1]
        
    z_logicals_aug = GF2(np.hstack([inner_products, z_logicals]))
    z_logicals_aug = z_logicals_aug.row_reduce(ncols=num_pairs)
    z_logicals = z_logicals_aug[:,num_pairs:]

    return z_logicals    

def get_logicals(partial_1 : sparse.spmatrix, partial_2 : sparse.spmatrix, compute_logicals, check_complex) -> QuantumCodeLogicals:
    x_logicals = np.zeros((0,partial_1.shape[1]), dtype=np.int8)
    z_logicals = np.zeros((0,partial_1.shape[1]), dtype=np.int8)
    if compute_logicals:
        partial_1_dense = GF2(partial_1.todense())
        partial_2_dense = GF2(partial_2.todense())

        x_logicals = compute_homology_reps(partial_2_dense, partial_1_dense)
        z_logicals = compute_homology_reps(partial_1_dense.T, partial_2_dense.T)
        z_logicals = compute_logical_pairs(z_logicals, x_logicals)
        
        if check_complex:
            for l in x_logicals:
                assert np.all((partial_1 @ l) % 2 == 0)
                
            for l in z_logicals:
                assert np.all((partial_2.T @ l) % 2 == 0)
            
    logicals = (
        [x_logicals[i,:] for i in range(x_logicals.shape[0])],
        [z_logicals[i,:] for i in range(z_logicals.shape[0])],
        len(x_logicals))
    return logicals

    

def homological_product(partial_A : sparse.spmatrix, partial_B : sparse.spmatrix, check_complex = None, compute_logicals = None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    '''Compute the homological product of two 2-complexes defined by their non-trivial boundary map
        Returns pair of boundary maps (partial_2, partial_1) of the total complex
    '''
    if check_complex is None:
        check_complex = False
        
    if compute_logicals is None:
        compute_logicals = False

    # D^A x I + I x D^B : A_1 x B_1 -> A_0 x B_1 + A_1 x B_0
    partial_2 = sparse.vstack([
        sparse.kron(partial_A, sparse.identity(partial_B.shape[1])),
        sparse.kron(sparse.identity(partial_A.shape[1]), partial_B)
    ]).astype(np.int8)
    
    partial_1_factors = [
        sparse.kron(sparse.identity(partial_A.shape[0]), partial_B),
        sparse.kron(partial_A, sparse.identity(partial_B.shape[0]))
    ]

    partial_1 = sparse.hstack(partial_1_factors).astype(np.int8)

    if check_complex:
        assert np.all((partial_1 @ partial_2).data % 2 == 0)

    logicals = get_logicals(partial_1, partial_2, compute_logicals, check_complex)

    # C2 dimension
    assert partial_2.shape[1] == partial_A.shape[1]*partial_B.shape[1]
    # C1 dimension
    assert partial_1.shape[1] == partial_A.shape[0]*partial_B.shape[1] + partial_A.shape[1]*partial_B.shape[0]
    assert partial_1.shape[1] == partial_2.shape[0]
    # C0 dimension
    assert partial_1.shape[0] == partial_A.shape[0]*partial_B.shape[0]

    assert len(logicals[0]) == len(logicals[1])

    # Check number of logicals + number of checks == number of qubits
    if compute_logicals:
        assert len(logicals[0]) + partial_2.shape[1] + partial_1.shape[0] == partial_2.shape[0]

    return ((partial_2.tocsc().astype(np.uint8), partial_1.tocsr().astype(np.uint8), num_cols(partial_1)), logicals)
