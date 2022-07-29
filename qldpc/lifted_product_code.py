from __future__ import annotations

from .homological_product_code import homological_product, get_logicals
import networkx as nx
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows
from .random_biregular_graph import random_biregular_graph
from .random_code import random_check_matrix
from .linalg import get_rank
import scipy.sparse as sparse
from itertools import product, chain
from collections import deque
import warnings

from typing import List, Set, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from galois import GF, FieldArray

class Group(ABC):
    @abstractmethod
    def __matmul__(self, other: Group) -> Group:
        pass

    @abstractmethod
    def inv(self) -> Group:
        pass

    @abstractmethod
    def identity(self) -> Group:
        pass

    @abstractmethod
    def __hash__(self):
        pass
    
    def __pow__(self, n:int) -> Group:
        '''Return g**n where n is a positive integer'''
        assert type(n) is int
        assert n >= 0
        r = self.identity()
        for _ in range(n):
            r = r @ self
        return r

# @dataclass(frozen=True)
class GL2(Group):
    '''GL(2,q)'''
    _gf : FieldArray
    data : FieldArray

    def __init__(self, gf : FieldArray, data : FieldArray):
        super().__init__()
        self._gf = gf
        self.data = gf(data)
        self.data.flags.writeable = False

    def __matmul__(self, other: GL2) -> GL2:
        return type(self)(self._gf, self.data @ other.data)

    def inv(self) -> GL2:
        return type(self)(self._gf, np.linalg.inv(self.data))

    def identity(self) -> GL2:
        return type(self)(self._gf, self._gf.Identity(2))

    def __hash__(self):
        return hash((self._gf.order, self.data.tobytes()))

    def __eq__(self, other):
        return (self._gf.order == other._gf.order) and np.all(self.data == other.data)

    def __repr__(self):
        return repr(self.data)


class PGL2(GL2):
    '''PGL(2,q) WIP. The quotient still needs to be implemented'''
    def __init__(self, gf : FieldArray, data : FieldArray, canonicalized=None):
        if canonicalized is None:
            canonicalized = False
        super().__init__(gf, data)
        # Create canonicalized matrix
        if not canonicalized:
            canonical_form = self.canonicalize()
            self.data = canonical_form.data
    
    def canonicalize(self) -> PGL2:
        '''Obtain a canonical representative of the element of PGL inside the coset of GL.
        We do this by forcing the top left entry to be 1. If it is zero then we force the top right element to be 1'''
        scaling = np.reciprocal(self.data[0,0] if self.data[0,0] != 0 else self.data[0,1])
        return type(self)(self._gf, self.data*scaling, canonicalized=True)

    def __matmul__(self, other: PGL2) -> PGL2:
        return type(self)(self._gf, super().__matmul__(other).data)

    def inv(self) -> PGL2:
        return type(self)(self._gf, super().inv().data)

    def identity(self) -> PGL2:
        return type(self)(self._gf, self._gf.Identity(2), canonicalized=True)

class Zqm(Group):
    '''The abelian group Z_q^m'''
    q : int
    m : int
    data : FieldArray

    def __init__(self, q : int, m : int, data : np.array) -> None:
        super().__init__()
        self.q = q
        self.m = m
        assert data.shape == (m,)
        assert np.all(data >= 0)
        assert np.all(data < q)
        assert np.issubdtype(data.dtype, int)
        self.data = data.astype(np.int_)
        self.data.flags.writeable = False

    def __matmul__(self, other: Zqm) -> Zqm:
        assert self.q == other.q
        return type(self)(self.q, self.m, (self.data + other.data)%self.q)

    def inv(self) -> Zqm:
        return type(self)(self.q, self.m, (self.q-self.data)%self.q)

    def identity(self) -> Zqm:
        return type(self)(self.q, self.m, np.zeros(self.m, dtype=np.int_))

    def __hash__(self):
        return hash((self.m, self.q, self.data.tobytes()))

    def __eq__(self, other):
        return (self.m == self.m) and (self.q == self.q)  and np.all(self.data == other.data)

    def __repr__(self):
        return repr(self.data)
    
def random_abelian_generators(q, m, k, symmetric=None, seed=None):
    '''Construct k generators (not necessarily complete or independent) at random for Z_q^m
    If symmetric is true, construct k/2 generators and add their inverses'''
    rng = np.random.default_rng(seed)
    # Rows of this matrix are desired generators wrt standard generators of Z_q^m
    if symmetric is None:
        symmetric = False
        
    # q = 2 is already symmetric
    symmetrize = symmetric and q != 2
        
    if symmetrize:
        if k%2 != 0:
            raise ValueError('Number of generators must be even when the set is symmetrized and q /= 2')
        k = k//2
        
    generator_matrix = rng.integers(low=0, high=q, size=(k,m))
    generators = [Zqm(q, m, generator_matrix[i,:]) for i in range(k)]
    if symmetrize:
        generators = list(chain(*[[g, g.inv()] for g in generators]))
    return generators
    
def test_random_abelian_generators():
    q = 3
    m = 4
    k = 5
    generators = random_abelian_generators(q,m,k, seed=42)
    group = _dfs_generators(generators[0].identity(), generators)
    assert len(group) == q**m

def morgenstern_generators(l, i) -> List[PGL2]:
    '''Construct the Morgenstern generators for PGL(2,q^i) with q = 2^l
    This follows the overview in Dinur et al. (2021) arXiv:2111.04808
    '''
    assert l >= 1
    # # This restriction is required by the text
    # assert l % 2 == 0
    q = 2**l
    Fq = GF(q)
    Fqi = GF(q**i)

    # We need to find some solutions, so we'll just exhaustively search for them
    # Find i \notin F_q s.t. i^2 + i \in F_q
    i_element = next(filter(lambda x: (x >= q) and (x**2 + x < q), Fqi.elements))
    eps = Fq(i_element**2 + i_element)

    # Find solutions to g^2 + gd + d^2 epsilon = 1
    def poly2(x):
        g,d = x
        return (g*g + g*d + d*d * eps == 1)
    map_Fqi = lambda x: tuple(map(Fqi, x))
    pairs = list(map(map_Fqi, filter(poly2, product(Fq.elements, Fq.elements))))
    assert len(pairs) == q+1
    x = Fqi.primitive_element # Is this right?
    generators = [PGL2(Fqi, [[1, (g+d*i_element)],[x*(g+d+d*i_element), 1]]) for (g,d) in pairs]
    return generators

def test_morgenstern_generators():
    l = 1
    i = 2
    generators = morgenstern_generators(l,i)
    identity = generators[0].identity()
    assert len(generators) == 2**l + 1
    
    group_elements = _dfs_generators(identity, generators)
    q = (2**l)**i
    assert len(group_elements) == (q-1)*q*(q+1)
    # Do DFS using the generators from the left and from the right to make sure we get the number of elements we expect
    # Check a \in A implies a^-1 \in A

def _dfs_generators(root : Group, generators : List[Group], traverse=None) -> Set[Group]:
    '''DFS traversal of the group from root using supplied generators acting from the left. A custom multiplication can be provided by passing traverse'''
    if traverse is None:
        traverse = lambda a,b: a@b
    
    visited = set()
    frontier = deque([root])
    while True:
        try:
            # Grab the next leaf on the frontier
            leaf = frontier.pop()
            if leaf in visited:
                continue
            visited.add(leaf)
        except IndexError:
            break
        # Compute the new fronter
        for g in generators:
            frontier.append(traverse(leaf, g))
            
    return visited

@dataclass(frozen=True)
class EdgeEdge:
    e: int
    g: Group
    f: int

@dataclass(frozen=True)
class VertexVertex:
    u: (int, int)
    g: Group
    v: (int, int)

@dataclass(frozen=True)
class EdgeVertex:
    e: int
    g: Group
    v: (int, int)

@dataclass(frozen=True)
class VertexEdge:
    v: (int, int)
    g: Group
    e: int

    
def lifted_product_code(group : List[Group], gen : List[Group], h1, h2, check_complex = None, compute_logicals = None, double_cover = None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    '''
    group object must implement __mul__ and inv()
        
    E x V -> ExE + VxV -> V x E
    h1 is the local system for the left factor
    h2 is the local system for the right factor
        
    The left factor group action is from the left
    The right factor group action is from the right

    The generator set will be extended to contain inverses in the construction
    
    double_cover=False uses the Bouquet graph as the base graph so the Tanner code is on the Cayley graph
    If double_cover is false we must also have that forall a \\in S, a /= a^-1
    
    double_cover=True uses a multigraph with w edges and 2 vertices so the Tanner code is on the double cover of the Cayley graph
    '''

    warnings.warn('Lifted Product codes is experimental!')
    
    if check_complex is None:
        check_complex = False

    if compute_logicals is None:
        compute_logicals = False

    if double_cover is None:
        double_cover = True

    w = len(gen)
    assert w == h1.shape[1]
    assert w == h2.shape[1]

    vertices = [0,1] if double_cover else [0]
    edge_boundaries = [(0,0), (1,1)] if double_cover else [(0,0), (0,1)]
    # we need to track orientation more carefully without the double cover
    vertex_coboundary = lambda v: [(e,v) for e in edges] if double_cover else [(e,o) for e in edges for o in (0,1)]
    
    edges = list(range(w))

    h1_system = list(range(h1.shape[0]))
    h2_system = list(range(h2.shape[0]))

    # Support of X check
    x_supports = dict()
    for (e1, g, v2, r2) in product(edges, group, vertices, h2_system):
        x_check = deque()
        # ExV -> VxV
        # Remembering the local system for E -> V
        # v1 is either 0 or 1 so we abuse notation and raise the shift to it
        x_check.extend(
            VertexVertex((v1, r1), gen[e1]**orient @ g, (v2, r2))
            for r1 in h1_system for v1, orient in edge_boundaries if h1[r1,e1] != 0)

        # ExV -> ExE
        x_check.extend(
            EdgeEdge(e1, g @ gen[e2].inv()**orient, e2)
            for e2, orient in vertex_coboundary(v2) if h2[r2,e2] != 0)

        x_supports[EdgeVertex(e1,g,(v2,r2))] = x_check
    
    # Supports of each qubit within a Z check
    q_supports = dict()
    # ExE -> VxE
    for (e1, g, e2) in product(edges, group, edges):
        support = deque()
        support.extend( VertexEdge((v1, r1), gen[e1]**orient @ g, e2)
                        for r1 in h1_system if h1[r1, e1] != 0 for v1, orient in edge_boundaries)

        q_supports[EdgeEdge(e1,g,e2)] = support

    # VxV -> VxE
    for (v1, r1, g, v2, r2) in product(vertices, h1_system, group, vertices, h2_system):
        support = deque()
        support.extend( VertexEdge((v1, r1), g @ gen[e2].inv()**orient, e2) for e2, orient in vertex_coboundary(v2) if h2[r2,e2] != 0)
        
        q_supports[VertexVertex((v1, r1), g, (v2, r2))] = support

    # Create indices for everything
    swap = lambda x: (x[1],x[0])
    x_check_indices = dict(map(swap, enumerate(x_supports.keys())))
    qubit_indices = dict(map(swap, enumerate(q_supports.keys())))
    z_check_indices = dict(map(swap, enumerate(VertexEdge((v1, r1), g, e2) for (v1, r1, g, e2) in product(vertices, h1_system, group, edges))))

    def coo_entries(coords):
        I = [x[0] for x in coords]
        J = [x[1] for x in coords]
        return (np.ones_like(I), (I,J))
    
    # Create boundary maps
    partial_2 = sparse.coo_matrix(
        coo_entries([(qubit_indices[qubit], x_check_indices[x_check])
                     for (x_check, x_check_support) in x_supports.items() for qubit in x_check_support]),
        shape=(len(qubit_indices), len(x_check_indices)), dtype=np.int32).tocsr()
    partial_1 = sparse.coo_matrix(
        coo_entries([(z_check_indices[z_check], qubit_indices[qubit])
                     for (qubit, qubit_support) in q_supports.items() for z_check in qubit_support]),
        shape=(len(z_check_indices), len(qubit_indices)), dtype=np.int32).tocsr()

    # In case of redundancies
    partial_2.data = partial_2.data%2
    partial_1.data = partial_1.data%2
    
    # Check complex and compute logical operators
    if check_complex:
        assert np.all((partial_1 @ partial_2).data % 2 == 0)

    checks = QuantumCodeChecks(partial_2.T.astype(np.int32), partial_1.astype(np.int32), partial_1.shape[1])
    logicals = get_logicals(checks, compute_logicals=compute_logicals, check_complex=check_complex)

    # dimensions match
    assert checks.x.shape[1] == checks.z.shape[1]
    assert len(logicals.x) == len(logicals.z)

    return (checks, logicals)

def _lifted_product_code_wrapper(generators, r, compute_logicals, seed, check_complex, r2=None, double_cover=None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    '''Utility function to reuse code between various LP code constructions'''
    assert r > 0
    r1 = r
    if r2 is None:
        r2 = r1

    if compute_logicals is None:
        compute_logicals = True

    if check_complex is None:
        check_complex = False

    w = len(generators)
    group = _dfs_generators(generators[0].identity(), generators)
    h1 = random_check_matrix(r1, w, seed=seed+1 if seed is not None else None)
    h2 = random_check_matrix(r2, w, seed=seed+2 if seed is not None else None)
    return lifted_product_code(group, generators, h1, h2, check_complex = check_complex, compute_logicals = compute_logicals, double_cover=double_cover)

def lifted_product_code_cyclic(q, m, w, r, compute_logicals=None, r2=None, seed=None, check_complex=None, double_cover=None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    '''Construct a lifted product code with w generators picked at random for Z_q^m. 
    The local systems of the left and right factors contains r constraints.
    If r2 is supplied then one factor in the lifted product will have r constraints and the other will have r2 constraints.
    The seed for each step will be derived from the passed seed if provided.
    '''
    assert q > 0
    assert m > 0
    assert w > 0

    if double_cover is None:
        double_cover = False

    if not double_cover:
        if q == 2:
            raise ValueError('Generators cannot be self inverse when not using the double cover')
        if w % 2 != 0:
            raise ValueError('Need an even degree for the Cayley graph when not using the double cover')
        w = w // 2
        
    generators = random_abelian_generators(q, m, w, seed=seed)
    return _lifted_product_code_wrapper(generators, r, compute_logicals=compute_logicals, r2=r2, seed=seed, check_complex=check_complex, double_cover=double_cover)

def lifted_product_code_pgl2(l, i, r, *args, **kwargs):
    '''Construct a lifted product code using the Morgenstern generators for PGL(2, (2^l)^i).
    Seed lifted_product_code_cyclic for other parameters.
    '''
    generators = morgenstern_generators(l, i)
    return _lifted_product_code_wrapper(generators, r, *args, **kwargs)

def test_lifted_product_code_cyclic():
    # Parameters from Higgot and Breuckmann
    w = 14
    r = 5
    q = 22
    m = 1
    G = q**m
    checks, logicals = lifted_product_code_cyclic(q=q, m=m, w=w, r=r, double_cover=True, compute_logicals=True, seed=42, check_complex=True)
    assert checks.num_qubits == (w**2 + 4*r**2)*G
    assert logicals.x.shape[0] >= checks.num_qubits - 2* (2*w*r*G)

def test_lifted_product_code_cyclic_Bw():
    # Don't use the double cover
    # Parameters from Higgot and Breuckmann
    w = 14
    r = 5
    q = 22
    m = 1
    G = q**m
    checks, logicals = lifted_product_code_cyclic(q=q, m=m, w=w, r=r, double_cover=False, compute_logicals=True, seed=42, check_complex=True)
    assert checks.num_qubits == (w**2//4 + r**2)*G
    assert logicals.x.shape[0] >= checks.num_qubits - w*r*G
    
def test_lifted_product_code_pgl2():
    # The local code length is probably too short here
    # TODO: Combine with a second set of generators
    checks, logicals = lifted_product_code_pgl2(1, 2, 5, compute_logicals=True, seed=42, check_complex=True)
    
