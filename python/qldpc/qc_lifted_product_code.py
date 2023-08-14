from .homological_product_code import homological_product, get_logicals
from .qecc_util import QuantumCode, QuantumCodeChecks
from galois import Poly, GF2
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import warnings

def shifts_to_polynomials(a):
    '''Given a matrix of shifts (powers of x) represented as integers, returns a matrix of polynomails suitable for passing to the code generator
    '''
    def scalar_to_poly(s):
        return Poly.Degrees([s])
    return np.vectorize(scalar_to_poly)(a)

def qc_lifted_product_code(quasicyclic_check_matrix, l, check_complex=None, compute_logicals=None) -> QuantumCode:
    '''
    Returns the quasi-cyclic lifted product code as defined in arxiv:2012.04068 using a matrix representation of the cyclic group.
    The input matrix is an n x m quasicyclic check matrix with elements in the polynomial ring GF2[x]/(x^l-1).
    If the input is integers, it will be interpreted as shifts i.e. the entry k will be replaced with x^k.
    '''
    if quasicyclic_check_matrix.dtype != 'O':
        quasicyclic_check_matrix = shifts_to_polynomials(quasicyclic_check_matrix)
    
    if check_complex is None:
        check_complex = False
        
    if compute_logicals is None:
        compute_logicals = False

    def circulant_reduce(a):
        return a % np.array(Poly.Degrees([l,0]))
        
    def identity(size):
        return np.identity(size, dtype=np.uint32)*np.array(Poly.One())
        
    def antipode(a):
        return Poly.Degrees((l-a.nonzero_degrees)%l)

    def poly_to_matrix(a):
        return linalg.circulant(a.coefficients(size=l, order='asc').astype(np.uint32))
    
    def embed_binary_matrix(a):
        a_blocks = [[poly_to_matrix(x) for x in row] for row in circulant_reduce(a)]
        return np.asarray(np.block(a_blocks))
    

    partial_A = quasicyclic_check_matrix
    partial_B = np.vectorize(antipode)(np.transpose(quasicyclic_check_matrix))

    # D^A x I + I x D^B : A_1 x B_1 -> A_0 x B_1 + A_1 x B_0
    partial_2 = embed_binary_matrix(np.vstack([
        np.kron(partial_A, identity(partial_B.shape[1])),
        np.kron(identity(partial_A.shape[1]), partial_B)
    ]))
    
    partial_1 = embed_binary_matrix(np.hstack([
        np.kron(identity(partial_A.shape[0]), partial_B),
        np.kron(partial_A, identity(partial_B.shape[0]))
    ]))

    if check_complex:
        assert np.all(np.asarray((partial_1 @ partial_2).data) % 2 == 0)

    checks = QuantumCodeChecks(sparse.csc_matrix(partial_2).transpose().astype(np.uint32), sparse.csr_matrix(partial_1).astype(np.uint32))
    logicals = get_logicals(checks, compute_logicals, check_complex)
    code = QuantumCode(checks, logicals)
    
    assert len(logicals.x) == len(logicals.z)
    assert checks.x.shape == checks.z.shape
    return code
