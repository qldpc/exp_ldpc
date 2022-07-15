from .homological_product_code import homological_product
import networkx as nx
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows
from .random_biregular_graph import random_biregular_graph
from .linalg import get_rank
import scipy.sparse as sparse
from itertools import product
from collections import deque

from dataclasses import dataclass
from abc import ABCMeta, abstractmethod, classmethod


class Group(ABCMeta):
    @classmethod
    @abstractmethod
    def __mul__(self: Group, other: Group) -> Group:
        pass

    @classmethod
    @abstractmethod
    def inv(self: Group) -> Group:
        pass

    @classmethod
    @abstractmethod
    def identity(self: Group) -> Group:
        pass

    def pow(self: Group, x:int) -> Group:
        '''Return g**x where x \in {0,1} i.e. identity or x'''
        assert x == 0 or x == 1
        return self if x == 1 else self.identity()

@dataclass
class EdgeEdge:
    e: int
    g: Group
    f: int

@dataclass
class VertexVertex:
    u: (int, int)
    g: Group
    v: (int, int)

@dataclass
class EdgeVertex:
    e: int
    g: Group
    v: (int, int)

@dataclass
class VertexEdge:
    v: (int, int)
    g: Group
    e: int



def lifted_product_code(group, gen, h1, h2, check_complex = None, compute_logicals = None) -> (QuantumCodeChecks, QuantumCodeLogicals):
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
        x_check.extend(
            VertexVertex((v1, r1), gen(e1).pow(v1) * g, (v2, r2))
            for r1 in h1_system for v1 in vertices if h1[r1,e1] != 0)

        # ExV -> ExE
        x_check.extend(
            EdgeEdge(e1, g * gen(e2).inv().pow(v2), e2)
            for e2 in edges if h2[r2,e2] != 0)

        x_supports[EdgeVertex(e1,g,(v2,r2))] = x_check
    
    # Supports of each qubit within a Z check
    q_supports = dict()
    for (e1, g, e2) in product(edges, group, edges):
        support = deque()
        support.extend( VertexEdge((v1, r1), gen(e1).pow(v1)*g, e2)
            for r1 in h1_system if h1[h1_system, e1] != 0 for v1 in vertices)

        q_supports[EdgeEdge(e1,g,e2)] = support

    for (v1, r1, g, v2, r2) in product(vertices, h1_system, group, vertices, h2_system):
        support = deque()
        support.extend( VertexEdge((v1, r1), g*gen(e2).inv().pow(v2), e2) for e2 in edges if h2[r2,e2] != 0)
        
        q_supports[VertexVertex((v1, r1), g, (v2, r2))] = support

    # Create indices for everything
    swap = lambda (x,y): (y,x)
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


