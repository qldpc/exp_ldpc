from .homological_product_code import homological_product, get_logicals
from .qecc_util import QuantumCode, QuantumCodeChecks
from .lifted_product_code import Group
from galois import Poly, GF2, FieldArray
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import warnings
from typing import Dict, List
    
class GroupAlgebra:
    scalar_field : FieldArray
    _data : Dict[Group, FieldArray]
  
    def __init__(self, gf : FieldArray, data : Dict[Group, FieldArray]):
        self.scalar_field = gf
        self._data = data
        self.canonicalize()
    
    def __matmul__(self, other):
        assert self.scalar_field == other.scalar_field
        return GroupAlgebra(self.scalar_field, {(a@b):v*u for (a,v) in self._data.items() for (b,u) in other._data.items()})
  
    def __add__(self, other):
        zero = self.scalar_field(1)
        keys = frozenset(self._data.keys()) | frozenset(other._data.keys())
        return GroupAlgebra(self.scalar_field, {k:(self.get(k, zero) + other.get(k, zero)) for k in keys})
  
    def antipode(self):
        '''Antipode map that takes each basis element to its multiplicative inverse'''
        return GroupAlgebra(self.scalar_field, {a.inv():u for (a,b) in self._data.items()})
    
    def canonicalize(self):
        self._data = dict(filter(lambda x: x != self.scalar_field(0), self._data.items()))

    def terms(self):
        '''Returns a dict of nonzero entries and their coefficient'''
        return copy(self._data)

def group_algebra_zero(gf):
    return GroupAlgebra(gf, {})
    
def group_algebra_monomial(scale : FieldArray, element : Group):
    return GroupAlgebra(scale, {element:scale})


class RegularRep:
    '''Class to construct the regular representation of a group with memoization. Defaults to a left regular representation'''
    _group : List[Group]
    _group_indices : Dict[Group, int]
    _matrices : Dict[Group, FieldArray]
    _field : FieldArray
    _left_action : bool
    
    def __init__(self, group : List[Group], field = None, left_action = None):
        self._group = group
        self._group_indices = dict(map(lambda x: (x[1],x[0]), enumerate(self._group)))

        if field is None:
            field = GF2
        self._field = field
        
        if left_action is None:
            left_action = True
        self._left_action = left_action
        self._matrices = {}

    def get_repp(self, element : Group):
        '''Returns a 0/1 matrix representation of the given element with entries in a finite field'''
        if element not in self._matrices:
            # Construct element lazily
            n = len(self._group)
            mat = self._field.Zeros((n,n))
            for g in _group:
                h = element @ g if self._left_action else g @ element
                mat[self._group_indices[h], self._group_indices[g]] = self._field(1)
            self._matrices[element] = mat

        return self._matrices[element]


def matrix_lifted_product_code(group, base_matrix, check_complex=None, compute_logicals=None) -> QuantumCode:
    '''
    Returns a lifted product code constructed as a lift of a base matrix.
    The input matrix is an n x m check matrix with elements in a group algebra.
    '''

    assert not base_matrix[0,0].scalar_field.is_extension_field
    assert base_matrix[0,0].scalar_field.characteristic == 2
    
    if check_complex is None:
        check_complex = False
        
    if compute_logicals is None:
        compute_logicals = False
        
    base_matrix = np.array(base_matrix)
    representation = RegularRep(group)
    
    def identity(size):
        return np.identity(size, dtype=np.uint32)*np.array(Poly.One())

    def group_alg_to_matrix(a):
        return sum(map(lambda x: x[1]*representation.get_rep(x[0]), a.terms().items()))
    
    def embed_binary_matrix(a):
        a_blocks = [[group_alg_to_matrix(x) for x in row] for row in a]
        return np.asarray(np.block(a_blocks))

    
    partial_A = base_matrix
    partial_B = np.vectorize(lambda x: x.antipode())(np.transpose(base_matrix))

    # D^A x I + I x D^B : A_1 x B_1 -> A_0 x B_1 + A_1 x B_0
    partial_2 = embed_binary_matrix(np.vstack([
        np.kron(partial_A, identity(partial_B.shape[1])),
        np.kron(identity(partial_A.shape[1]), partial_B)
    ]))
    
    partial_1 = embed_binary_matrix(np.hstack([
        np.kron(identity(partial_A.shape[0]), partial_B),
        np.kron(partial_A, identity(partial_B.shape[0]))
    ]))

    # The rest of the package works with integer matrices
    partial_1 = np.array(partial_1, dtype=np.uint8)
    partial_2 = np.array(partial_2, dtype=np.uint8)

    if check_complex:
        assert np.all(np.asarray((partial_1 @ partial_2).data) % 2 == 0)

    checks = QuantumCodeChecks(sparse.csc_matrix(partial_2).transpose().astype(np.uint32), sparse.csr_matrix(partial_1).astype(np.uint32))
    logicals = get_logicals(checks, compute_logicals, check_complex)
    code = QuantumCode(checks, logicals)
    
    assert len(logicals.x) == len(logicals.z)
    assert checks.x.shape == checks.z.shape
    return code
