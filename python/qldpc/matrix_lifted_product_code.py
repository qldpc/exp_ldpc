from .homological_product_code import homological_product, get_logicals
from .qecc_util import QuantumCode, QuantumCodeChecks
from .lifted_product_code import Group
from galois import Poly, GF2, FieldArray, GF
from galois.typing import ElementLike
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
import warnings
from functools import reduce
from typing import Dict, List
from copy import deepcopy

class GroupAlgebra:
    scalar_field : FieldArray
    _data : Dict[Group, FieldArray]
  
    def __init__(self, gf : FieldArray, data : Dict[Group, FieldArray]):
        self.scalar_field = gf
        self._data = data
        self.canonicalize()
    
    def _mul_GA(self, other):
        assert self.scalar_field == other.scalar_field
        output_dict = {}
        zero = self.scalar_field.Zeros(1)[0]
        for (a,v) in self._data.items():
          for (b,u) in other._data.items():
            c = a@b
            output_dict[c] = output_dict.get(c,zero) + v*u
        return GroupAlgebra(self.scalar_field, output_dict)

    def _mul_scalar(self,other):
        return GroupAlgebra(self.scalar_field, {a:u*other for (a,u) in self._data.items()})
    
    def __mul__(self, other : ElementLike):
        if isinstance(other,GroupAlgebra):
            return self._mul_GA(other)
        else:
            return self._mul_scalar(other)
    
    def __add__(self, other):
        zero = self.scalar_field.Zeros(1)[0]
        keys = frozenset(self._data.keys()) | frozenset(other._data.keys())
        return GroupAlgebra(self.scalar_field, {k:(self._data.get(k, zero) + other._data.get(k, zero)) for k in keys})
  
    def antipode(self):
        '''Antipode map that takes each basis element to its multiplicative inverse'''
        return GroupAlgebra(self.scalar_field, {a.inv():u for (a,u) in self._data.items()})
    
    def canonicalize(self):
        zero = self.scalar_field.Zeros(1)[0]
        self._data = dict(filter(lambda x: x[1] != zero, self._data.items()))

    def terms(self):
        '''Returns a dict of nonzero entries and their coefficient'''
        return deepcopy(self._data)

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

    def zero(self):
        '''Returns a zero matrix of the same shape as the matrix representation'''
        n = len(self._group)
        return self._field.Zeros((n,n))

    def get_rep(self, element : Group):
        '''Returns a 0/1 matrix representation of the given element with entries in a finite field'''
        if element not in self._matrices:
            # Construct element lazily
            mat = self.zero()
            one = self._field.Ones(1)[0]
            for g in self._group:
                h = element @ g if self._left_action else g @ element
                mat[self._group_indices[h], self._group_indices[g]] = one
            self._matrices[element] = mat

        return self._matrices[element]

def matrix_lifted_product_code(group, base_matrix_A, base_matrix_B=None, dual_A=None, dual_B = None, check_complex=None, compute_logicals=None) -> QuantumCode:
    '''
    Returns a lifted product code constructed as a lift of a base matrix.
    The input matrix is an n x m check matrix with elements in a group algebra.
    The base matrix must be a group algebra over F2, but this is not enforced currently

    If B is not given, then we will use B = A*
    If dual_A or dual_B is pass, the transpose and antipode map will be applied

    Base matrices are the map defining length-1 chain complexes from which we will take the standard tensor product of chain complexes
    A: A1 -> A0
    B: B1 -> B0
    '''

    # assert not base_matrix[0,0].scalar_field.is_extension_field
    # assert base_matrix[0,0].scalar_field.characteristic == 2

    # Defaults 
    if check_complex is None:
        check_complex = False
        
    if compute_logicals is None:
        compute_logicals = False

    
    if base_matrix_B is None:
        assert dual_B is None and dual_A is None
    if dual_A is None:
        dual_A = False
    if dual_B is None:
        dual_B = False

    #  -----
    # Map from provided base matrices to boundary operator of the chain complex
    # Optionally take dual of base matrices
    def dual(a):
        return np.vectorize(lambda x: x.antipode())(np.transpose(a))

    base_matrix_A = np.array(base_matrix_A)
    field_one = base_matrix_A[0,0].scalar_field.Ones(1)[0]
    
    partial_A = np.array(base_matrix_A)
    if base_matrix_B is not None:
        partial_B = np.array(base_matrix_B)
    else:
        partial_B = dual(partial_A)

    if dual_A:
        partial_A = dual(partial_A)
    if dual_B:
        partial_B = dual(partial_B)

    #  -----
    # Construct representation as permutation matrices
    
    representation = RegularRep(group)
    
    def identity(size):
        group_id = group[0].identity()
        ga_one = group_algebra_monomial(field_one, group_id)
        return np.vectorize(lambda x: ga_one*x)(field_one.Identity(size))

    def group_alg_to_matrix(a):
        return sum(map(lambda x: x[1]*representation.get_rep(x[0]), a.terms().items()), representation.zero())
    
    def embed_binary_matrix(a):
        a_blocks = [[group_alg_to_matrix(x) for x in row] for row in a]
        return np.asarray(np.block(a_blocks))

    #  -----

    # Build complex
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
