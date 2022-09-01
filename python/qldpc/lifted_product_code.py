from __future__ import annotations
from multiprocessing.sharedctypes import Value

from .homological_product_code import get_logicals
import numpy as np
from .qecc_util import QuantumCode, QuantumCodeChecks
from .random_code import random_check_matrix
import scipy.sparse as sparse
from itertools import product, chain
from collections import deque
import warnings
import networkx as nx

from typing import List, Set, Tuple
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
    u: Tuple[int, int]
    g: Group
    v: Tuple[int, int]

@dataclass(frozen=True)
class EdgeVertex:
    e: int
    g: Group
    v: Tuple[int, int]

@dataclass(frozen=True)
class VertexEdge:
    v: Tuple[int, int]
    g: Group
    e: int

# networkx edges are unhashable because the weights are a dict
def _unpack_edge(edge : Tuple(int, int, dict)) -> Tuple(int, int, Group):
    return (edge[0], edge[1], edge[2]['g'], edge[2]['idx'])

def lifted_product_code(group : List[Group], generators : List[Group], h1, h2, check_complex = None, compute_logicals = None, double_cover = None) -> QuantumCode:
    '''
    group object must implement __mul__ and inv()
        
    E x V -> ExE + VxV -> V x E
    h1 is the local system for the left factor
    h2 is the local system for the right factor
        
    The left factor group action is from the left
    The right factor group action is from the right

    gen isomorphic to the number of outgoing edges in the base graph I.e. |gen| = w and base graph is B_w or D_w
    '''

    warnings.warn('Lifted Product codes is experimental!')
    
    if check_complex is None:
        check_complex = False

    if compute_logicals is None:
        compute_logicals = False

    if double_cover is None:
        double_cover = True

    if h1.shape[1] != h2.shape[1]:
        raise ValueError('Local code block lengths must match. (For now)')

    # Make base graph
    # We put an extra index i here to identify duplicate generators
    base_graph = nx.MultiDiGraph()
    if double_cover:
        base_graph.add_nodes_from([0,1])
        base_graph.add_edges_from((0,1,{'g':g,'idx':i}) for i, g in enumerate(generators))
    else:
        base_graph.add_node(0)
        base_graph.add_edges_from((0,0,{'g':g,'idx':i}) for i, g in enumerate(generators))

    # Indices that we will use to index into the local system
    # We need to seperately index in and out edges because an edge could be a self edge, so it would appear in the local system twice
    for v in base_graph.nodes:
        base_graph.nodes[v]['out_idx'] = {_unpack_edge(e):i for i, e in enumerate(base_graph.out_edges(v, data=True))}
        in_indices_offset = len(base_graph.nodes[v]['out_idx'])
        base_graph.nodes[v]['in_idx'] = {_unpack_edge(e):(i+in_indices_offset) for i, e in enumerate(base_graph.in_edges(v, data=True))}

        if len(base_graph.nodes[v]['out_idx']) + len(base_graph.nodes[v]['in_idx']) != h1.shape[1]:
            print(f"{len(base_graph.nodes[v]['out_idx']) + len(base_graph.nodes[v]['in_idx'])=}, {h1.shape[1]=}")
            raise ValueError('Local code block length does not match base graph degree')
    
    # local system indices
    h1_system = list(range(h1.shape[0]))
    h2_system = list(range(h2.shape[0]))

    # Supports of X checks
    x_supports = dict()
    for (e1, v2, r2, g) in product(base_graph.edges(data=True), base_graph.nodes, h2_system, group):
        x_check = deque()
        # ExV -> VxV
        # Edge endpoints
        u1,v1 = e1[:2]

        # ->(v1) of e1
        x_check.extend(
            VertexVertex((v1, r1), e1[2]['g'] @ g, (v2, r2))
            for r1 in h1_system if h1[r1,base_graph.nodes[v1]['in_idx'][_unpack_edge(e1)]] != 0)
        # (u1)-> of e1
        x_check.extend(
            VertexVertex((u1, r1), g, (v2, r2))
            for r1 in h1_system if h1[r1,base_graph.nodes[u1]['out_idx'][_unpack_edge(e1)]] != 0)

        # ExV -> ExE
        # Out edges
        x_check.extend(
            EdgeEdge(_unpack_edge(e1), g, _unpack_edge(e2))
            for e2 in base_graph.out_edges(v2, data=True) if h2[r2, base_graph.nodes[v2]['out_idx'][_unpack_edge(e2)]] != 0)
        # In edges 
        x_check.extend(
            EdgeEdge(_unpack_edge(e1), g @ e2[2]['g'].inv(), _unpack_edge(e2))
            for e2 in base_graph.in_edges(v2, data=True) if h2[r2, base_graph.nodes[v2]['in_idx'][_unpack_edge(e2)]] != 0)

        x_supports[EdgeVertex(_unpack_edge(e1),g,(v2,r2))] = x_check

    # Supports of each qubit within a Z check
    q_supports = dict()
    # ExE -> VxE
    for (e1, g, e2) in product(base_graph.edges(data=True), group, base_graph.edges(data=True)):
        support = deque()
        u1,v1 = e1[:2]
        support.extend(VertexEdge((v1, r1), e1[2]['g'] @ g, _unpack_edge(e2))
            for r1 in h1_system if h1[r1, base_graph.nodes[v1]['in_idx'][_unpack_edge(e1)]] != 0)

        support.extend(VertexEdge((u1, r1), e1[2]['g'] @ g, _unpack_edge(e2))
            for r1 in h1_system if h1[r1, base_graph.nodes[u1]['out_idx'][_unpack_edge(e1)]] != 0)

        q_supports[EdgeEdge(_unpack_edge(e1),g,_unpack_edge(e2))] = support

    # VxV -> VxE
    for (v1, r1, g, v2, r2) in product(base_graph.nodes, h1_system, group, base_graph.nodes, h2_system):
        support = deque()
        support.extend( VertexEdge((v1, r1), g, _unpack_edge(e2)) for e2 in base_graph.out_edges(v2, data=True) if h2[r2, base_graph.nodes[v2]['out_idx'][_unpack_edge(e2)]] != 0)
        support.extend( VertexEdge((v1, r1), g @ e2[2]['g'].inv(), _unpack_edge(e2)) for e2 in base_graph.in_edges(v2, data=True) if h2[r2, base_graph.nodes[v2]['in_idx'][_unpack_edge(e2)]] != 0)
        
        q_supports[VertexVertex((v1, r1), g, (v2, r2))] = support

    
    # Create indices for everything
    swap = lambda x: (x[1],x[0])
    x_check_indices = dict(map(swap, enumerate(x_supports.keys())))
    qubit_indices = dict(map(swap, enumerate(q_supports.keys())))
    z_check_indices = dict(map(swap, enumerate(VertexEdge((v1, r1), g, _unpack_edge(e2)) for (v1, r1, g, e2) in product(base_graph.nodes, h1_system, group, base_graph.edges(data=True)))))

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

    checks = QuantumCodeChecks(partial_2.T.astype(np.uint32), partial_1.astype(np.uint32))
    logicals = get_logicals(checks, compute_logicals=compute_logicals, check_complex=check_complex)

    # dimensions match
    assert checks.x.shape[1] == checks.z.shape[1]
    assert len(logicals.x) == len(logicals.z)

    return QuantumCode(checks, logicals)

def _lifted_product_code_wrapper(generators, r, compute_logicals, seed, check_complex, r2=None, double_cover=None) -> QuantumCode:
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
    h1 = random_check_matrix(r1, w if double_cover else w*2, seed=seed+1 if seed is not None else None)
    h2 = random_check_matrix(r2, w if double_cover else w*2, seed=seed+2 if seed is not None else None)
    return lifted_product_code(group, generators, h1, h2, check_complex = check_complex, compute_logicals = compute_logicals, double_cover=double_cover)

def lifted_product_code_cyclic(q, m, w, r, compute_logicals=None, r2=None, seed=None, check_complex=None, double_cover=None) -> QuantumCode:
    '''Construct a lifted product code with w generators picked at random for Z_q^m. 
    When using the double cover, the local code block length is twice the number of generators.
    The local systems of the left and right factors contains r constraints.
    If r2 is supplied then one factor in the lifted product will have r constraints and the other will have r2 constraints.
    The seed for each step will be derived from the passed seed if provided.
    '''
    assert q > 0
    assert m > 0
    assert w > 0

    if double_cover is None:
        double_cover = False
        
    generators = random_abelian_generators(q, m, w, seed=seed)
    return _lifted_product_code_wrapper(generators, r, compute_logicals=compute_logicals, r2=r2, seed=seed, check_complex=check_complex, double_cover=double_cover)

def lifted_product_code_pgl2(l, i, r, *args, **kwargs):
    '''Construct a lifted product code using the Morgenstern generators for PGL(2, (2^l)^i).
    When using the double cover, the local code block length is twice the number of generators.
    Seed lifted_product_code_cyclic for other parameters.
    '''
    generators = morgenstern_generators(l, i)
    return _lifted_product_code_wrapper(generators, r, *args, **kwargs)
    
