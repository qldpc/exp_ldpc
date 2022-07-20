from __future__ import annotations

from .homological_product_code import homological_product
import networkx as nx
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows
from .random_biregular_graph import random_biregular_graph
from .linalg import get_rank
import scipy.sparse as sparse
from itertools import product
from collections import deque

from typing import List
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
        assert n is int
        assert n >= 0
        r = self.identity()
        for _ in range(n):
            r = r * self
        return r

# @dataclass(frozen=True)
class GL2(Group):
    '''GL(2,q)'''
    _gf : FieldArray
    data : FieldArray

    def __init__(self, gf : FieldArray, data) -> None:
        super().__init__()
        self._gf = gf
        self.data = gf(data)
        self.data.flags.writeable = False

    def __matmul__(self, other: PGL2) -> PGL2:
        return type(self)(self._gf, self.data @ other.data)

    def inv(self) -> PGL2:
        return type(self)(self._gf, np.linalg.inv(self.data))

    def identity(self) -> PGL2:
        return type(self)(self._gf, self._gf.Identity(2))

    def __hash__(self):
        return hash((self._gf.order, self.data.tobytes()))

    def __eq__(self, other):
        return (self._gf.order == other._gf.order) and np.all(self.data == other.data)

    def __repr__(self):
        return repr(self.data)
    
class PGL2(GL2):
    '''PGL(2,q) WIP. The quotient still needs to be implemented'''
    pass

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
        return type(self)(self.q, self.m, (q-self.data)%self.q)

    def identity(self) -> Zqm:
        return type(self)(self.q, self.m, np.zeros(self.m, dtype=np.int_))

    def __hash__(self):
        return hash((self.m, self.q, self.data.tobytes()))

    def __eq__(self, other):
        return (self.m == self.m) and (self.q == self.q)  and np.all(self.data == other.data)

    def __repr__(self):
        return repr(self.data)
    
def random_abelian_generators(q, m, k, seed=None):
    '''Construct k generators (not necessarily complete or independent) at random for Z_q^m'''
    rng = np.random.default_rng(seed)
    # Rows of this matrix are desired generators wrt standard generators of Z_q^m
    generator_matrix = rng.integers(low=0, high=q, size=(k,m))
    return [Zqm(q, m, generator_matrix[i,:]) for i in range(k)]

def test_random_abelian_generators():
    q = 3
    m = 4
    k = 5
    generators = random_abelian_generators(q,m,k, seed=42)
    group = _dfs_generators(lambda a,b: a@b, generators[0].identity(), generators)
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
    i_element = next(filter(lambda x: (x >= q) and (x**2 + x < q), Fqi.Elements()))
    eps = Fq(i_element**2 + i_element)

    # Find solutions to g^2 + gd + d^2 epsilon = 1
    def poly2(x):
        g,d = x
        return (g*g + g*d + d*d * eps == 1)
    map_Fqi = lambda x: tuple(map(Fqi, x))
    pairs = list(map(map_Fqi, filter(poly2, product(Fq.Elements(), Fq.Elements()))))
    assert len(pairs) == q+1
    x = Fqi.primitive_element # Is this right?
    generators = [PGL2(Fqi, [[1, (g+d*i_element)],[x*(g+d+d*i_element), 1]]) for (g,d) in pairs]
    return generators

def test_morgenstern_generators():
    l = 1
    i = 2
    generators = morgenstern_generators(l,i)
    identity = generators[0].identity()

    group_elements = _dfs_generators(lambda a,b: a@b, identity, generators)
    q = (2**l)**i
    assert len(group_elements) == (q-1)*q*(q+1)
    # Do DFS using the generators from the left and from the right to make sure we get the number of elements we expect
    # Check a \in A implies a^-1 \in A

def _dfs_generators(traverse, root, generators):
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

    
def lifted_product_code(group : List[Group], gen : List[Group], h1, h2, check_complex = None, compute_logicals = None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    ''' group object must implement __mul__ and inv()
        
        E x V -> ExE + VxV -> V x E
        h1 is the local system for the left factor
        h2 is the local system for the right factor
        
        The left factor group action is from the left as normal
        The right factor group action is from the right as an inverse
    '''

    w = len(gen)
    assert w == h1.shape[1]
    assert w == h2.shape[1]

    vertices = [0,1]
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
            VertexVertex((v1, r1), gen(e1)**v1 * g, (v2, r2))
            for r1 in h1_system for v1 in vertices if h1[r1,e1] != 0)

        # ExV -> ExE
        x_check.extend(
            EdgeEdge(e1, g * gen(e2).inv()**v2, e2)
            for e2 in edges if h2[r2,e2] != 0)

        x_supports[EdgeVertex(e1,g,(v2,r2))] = x_check
    
    # Supports of each qubit within a Z check
    q_supports = dict()
    for (e1, g, e2) in product(edges, group, edges):
        support = deque()
        support.extend( VertexEdge((v1, r1), gen(e1)**v1 *g, e2)
            for r1 in h1_system if h1[h1_system, e1] != 0 for v1 in vertices)

        q_supports[EdgeEdge(e1,g,e2)] = support

    for (v1, r1, g, v2, r2) in product(vertices, h1_system, group, vertices, h2_system):
        support = deque()
        support.extend( VertexEdge((v1, r1), g*gen(e2).inv()**v2, e2) for e2 in edges if h2[r2,e2] != 0)
        
        q_supports[VertexVertex((v1, r1), g, (v2, r2))] = support

    # Create indices for everything
    swap = lambda : (x[1],x[0])
    x_check_indices = dict(map(swap, enumerate(x_supports.keys())))
    qubit_indices = dict(map(swap, enumerate(q_supports.keys())))
    z_check_indices = dict(map(swap, enumerate(((v1, r1), g, e2) for (v1, r1, g, r2) in product(vertices, h1_system, group, edges))))

    def coo_entries(coords):
        I = [x[0] for x in coords]
        J = [x[1] for x in coords]
        return (1, (I,J))
    
    # Create boundary maps
    partial_2 = sparse.coo_matrix(
        coo_entries((qubit_indices(qubit), x_check_indices(x_check))
            for (x_check, x_check_support) in x_supports.items() for qubit in x_check_supp),
        shape=(len(qubit_indices), len(x_check_indices))).to_csr()
    partial_1 = sparse.coo_matrix(
        coo_entries((z_check_indices(z_check), qubit_indices(qubit))
            for (qubit, qubit_support) in q_supports.items() for z_check in qubit_support),
        shape=(len(z_check_indices), len(qubit_indices))).to_csr()

    # Check complex and compute logical operators
    if check_complex:
        assert np.all((partial_1 @ partial_2).data % 2 == 0)

    if compute_logicals is None:
        compute_logicals = False
    logicals = get_logicals(partial_1, partial_2, compute_logicals)

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


def test_lifted_product_code():
    pass


