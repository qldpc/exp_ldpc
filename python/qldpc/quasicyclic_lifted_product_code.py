from .homological_product_code import homological_product
from .qecc_util import QuantumCode
from galois import Poly, GF2
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import warnings

def circulant_generator(k, circulant_size):
    '''returns the l by l circulant matrix that implents the shift to the right by k positions'''
    return linalg.circulant([1 if i == k else 0 for i in range(circulant_size)])

def shifts_to_blocks(a, circulant_size):
    '''Given a matrix of shifts (powers of x) represented as integers, returns the full block matrix suitable for passing into the code generator
    '''
    pass

def quasicyclic_lifted_product_code(quasicyclic_check_matrix, circulant_size, check_complex=None, compute_logicals=None) -> QuantumCode:
    '''
    The input matrix is an n x m quasicyclic check matrix with elements in the polynomial ring GF2[x]/(x^l-1) represented as l by l permutation matrices over GF2.
    This is represented as an n x m x l x l tensor A_abcd in GF2 where the i,j coefficient (block) of the matrix is the lxl matrix with cd coefficient equal to A_ijcd
    '''
    
    if check_complex is None:
        check_complex = False
        
    if compute_logicals is None:
        compute_logicals = False

    
    # def block_repr(a):
    #     return linalg.circulant(a.cofficients(size=circulant_size, order='asc'))

    partial_A = quasicyclic_check_matrix.astype(np.uint32)
    partial_B = np.einsum('abcd->badc', quasicyclic_check_matrix).astype(np.uint32)

    # Takes two matrices with the block structure and returns the kronecker product with block structure forgotten
    def kronecker_embed(a, b):
        # Indices left to right are A, B, circulant repp
        assert a.shape[2] == circulant_size
        assert b.shape[2] == circulant_size
        product = np.einsum('abcd,efdh->aecbfh')%2
        return np.reshape(product, (a.shape[0]*b.shape[0]*circulant_size, a.shape[1]*b.shape[1]*circulant_size), order='F')

    def circulant_identity(size):
        inner_identity = np.identity(circulant_size, dtype=np.uint32)
        outer_identity = np.identity(size, dtype=np.uint32)
        return np.einsum('a,b->ab',outer_identity, inner_identity)

    # D^A x I + I x D^B : A_1 x B_1 -> A_0 x B_1 + A_1 x B_0
    partial_2 = np.vstack([
        kronecker_embed(partial_A, circulant_identity(partial_B.shape[1])),
        kronecker_embed(circulant_identity(partial_A.shape[1]), partial_B)
    ]).astype(np.uint32)
    
    partial_1_factors = [
        sparse.kron(circulant_identity(partial_A.shape[0]), partial_B),
        sparse.kron(partial_A, circulant_identity(partial_B.shape[0]))
    ]

    partial_1 = np.hstack(partial_1_factors).astype(np.uint32)

    if check_complex:
        assert np.all((partial_1 @ partial_2).data % 2 == 0)

    checks = QuantumCodeChecks(sparse.csc_matrix(partial_2).transpose().astype(np.uint32), sparse.csr_matrix(partial_1).astype(np.uint32))
    logicals = get_logicals(checks, compute_logicals, check_complex)
    code = QuantumCode(checks, logicals)
    
    assert len(logicals.x) == len(logicals.z)
    assert checks.x.shape == checks.z.shape
    assert checks.num_qubits == (num_data**2 + num_checks**2)
    return code
