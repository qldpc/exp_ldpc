import numpy as np
from scipy import sparse
import numpy.typing as npt
import networkx as nx
from .edge_coloring import edge_color_bipartite
from collections import deque
from typing import List,Tuple,Deque
import pytest

def product_permutation_route(R : npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    '''Compute a permutation routing on a graph isomorphic to some product G x H.
    G and H are indexed by integers [0, |G|) and [0, |H|) respectively.
    R is a |G|x|H|x2 array with R[g0,h0,:] = (g1,h1) denoting that (g0,h0) should be sent to (g1,h1).
    The return is an array A : G x H -> G.
    If R[g0,h0,:] = (g1,h1), the returned array A[g0,h0] = g indicates that the element should first be routed to (g,h0) then (g,h1) then (g1,h1).
    I.e. for two intervals, it looks like

    h0
 g0 o
    |
    |      o g1
    |      |
  g o------o
           h1

    Combine this with routing algorithms for G and H to have a full permutation routing

    Algo is from
    M. Baumslag and F. Annexstein, Math. Systems Theory 24, 233-251 (1991).
    '''
    
    G_size = R.shape[0]
    H_size = R.shape[1]
    
    assert R.shape == (G_size, H_size, 2)
    # Check that R is a valid permutation
    # Indices in range
    for i in range(G_size):
        for j in range(H_size):
            assert 0 <= R[i,j,0] and R[i,j,0] < G_size
            assert 0 <= R[i,j,1] and R[i,j,1] < H_size
    # Destinations unique
    R_entries = [tuple(R[i,j,:]) for i in range(G_size) for j in range(H_size)]
    assert len(R_entries) == len(set(R_entries))
    
    # Bipartite graph with vertex set H \sqcup H
    # An edge (h0, h1) exists iff there is some g s.t. R[g,h0,1] = h1
    # I.e. we put an edge between two columns if the left column contains an element that needs to go to the right column
    col_graph_edge_list = [(h0, H_size + R[g0, h0, 1], {'g0':g0}) for g0 in range(G_size) for h0 in range(H_size)]
    col_graph = nx.MultiGraph()
    col_graph.add_nodes_from(range(H_size), bipartite=0)
    col_graph.add_nodes_from(range(H_size,2*H_size), bipartite=1)
    col_graph.add_edges_from(col_graph_edge_list)

    # This is a decomposition into perfect matchings

    # We route along a different column for each color
    edge_coloring = edge_color_bipartite(col_graph)

    A = np.zeros((G_size, H_size))
    for coloring, edgeset in enumerate(edge_coloring):
        for edge in edgeset:
            h0 = edge[0]
            g0 = col_graph.edges[edge]['g0']
            A[g0,h0] = coloring
    return A


def assert_disjoint_swaps(swap_schedule):
    # Check that all swaps in the same round are disjoint
    for round in swap_schedule:
        round_set = set()
        for swap_op in round:
            for target in swap_op:
                assert target not in round_set
                round_set.add(target)

def execute_swaps(a, swap_schedule):
    for round in swap_schedule:
        for swap_op in round:
            t = np.copy(a[swap_op[0][0],swap_op[0][1],:])
            a[swap_op[0][0],swap_op[0][1],:] = a[swap_op[1][0],swap_op[1][1],:]
            a[swap_op[1][0],swap_op[1][1],:] = t

def _even_odd_sort_inplace(interval, compare, swap):
    '''Even / odd neighest neighbor sorting network
    Uses the compare and swap functions to sort in the interval [0,interval).
    Returns a list of lists of swaps performed at each timestep
    '''
    
    swap_list = deque()
    for n in range(interval):
        swap_timestep = deque()
        index_set = range(0, interval-1, 2) if n % 2 == 0 else range(1, interval-1, 2)
        for i in index_set:
            if not compare(i, i+1):
                swap_timestep.append(swap(i, i+1))
        swap_list.append(swap_timestep)
    return swap_list

def _collate_swaps(row_swaps):
    # combine a list containing for each row: A list containing the list of swaps to apply at each timestep
    combined = deque()
    if len(row_swaps) == 0:
        return combined
    
    # Check that all the lengths match
    assert len(frozenset([len(r) for r in row_swaps])) == 1
    
    while len(row_swaps[0]) > 0:
        timestep = deque()
        for row in row_swaps:
            timestep.extend(row.popleft())            
        combined.append(timestep)
    return combined


Swap = Tuple[Tuple[int, int], Tuple[int, int]]
def grid_permutation_route(R : npt.NDArray[np.int_]) -> List[Deque[Swap]]:
    '''Finds a sequence of nearest-neighbor swaps on a grid that implements the desired routing.
    The input specifies the permutation: R[i0,j0, :] = [i1,j1] indicates that the qubit at (i0,j0) should be routed to (i1,j1).
    The output is a list of deques of swaps to be performed.
    All the swaps in each entry of the list can be parallelized.
    '''
    

    G_size = R.shape[0]
    H_size = R.shape[1]
    routing_row = np.reshape(product_permutation_route(R), (G_size, H_size, 1))
    
    # Compute the route
    route = np.concatenate([R, routing_row], axis=2)
    swaps = []

    def route_stage(is_rows, key):
        A = route.transpose((1,0,2)) if is_rows else route.view()
        # route along the second axis of A
        stage_swaps = deque()
        for i in range(A.shape[0]):
            compare = lambda j0,j1: A[i,j0,key] <= A[i,j1,key]
            def swap(j0, j1):
                A[i,[j1,j0],:] = A[i,[j0,j1],:]
                return ((j0,i),(j1,i)) if is_rows else ((i, j0),(i,j1))
            stage_swaps.append(_even_odd_sort_inplace(A.shape[1], compare, swap))
        swaps.extend(_collate_swaps(stage_swaps))

    # Route along cols to routing row
    # Then route to destination column
    # Then route to destination row
    route_stage(True, 2)
    route_stage(False, 1)
    route_stage(True, 0)

    return swaps

def _random_permutation(G_size, H_size):
    permutation = np.array([(i,j) for i in range(G_size) for j in range(H_size)])
    rng = np.random.default_rng(seed=30)
    rng.shuffle(permutation)
    permutation = np.reshape(permutation, (G_size, H_size, 2))
    return permutation

HG_sizes = [(11,7), (10, 5), (6, 8), (6, 9)]

@pytest.mark.parametrize('G_size,H_size', HG_sizes)
def test_product_permutation_route(G_size, H_size):
    # Test that the routing returned by product permutation route is congestion-free

    for _ in range(100):
        permutation = _random_permutation(G_size, H_size)
        routing_row = np.reshape(product_permutation_route(permutation), (G_size, H_size, 1))
        
        # Compute the route
        route = np.concatenate([permutation, routing_row], axis=2)

        # Route along cols
        for j in range(H_size):
            col = [tuple(route[i,j,:]) for i in range(G_size)]
            col.sort(key=lambda x: x[2])
            for i in range(G_size):
                route[i,j,:] = col[i]

        # Route along rows
        for i in range(G_size):
            row = [tuple(route[i,j,:]) for j in range(H_size)]
            row.sort(key=lambda x: x[1])
            for j in range(H_size):
                route[i,j,:] = row[j]

        # Route along columns
        for j in range(H_size):
            col = [tuple(route[i,j,:]) for i in range(G_size)]
            col.sort(key=lambda x: x[0])
            for i in range(G_size):
                route[i,j,:] = col[i]

        # check
        for i in range(G_size):
            for j in range(H_size):
                assert tuple(route[i,j,:2]) == (i,j)

@pytest.mark.parametrize('G_size,H_size', HG_sizes)
def test_grid_permutation_route(G_size, H_size):

    for _ in range(100):
        permutation = _random_permutation(G_size, H_size)
        swap_schedule = grid_permutation_route(np.copy(permutation))
        assert_disjoint_swaps(swap_schedule)
        execute_swaps(permutation, swap_schedule)
        for i in range(G_size):
            for j in range(H_size):
                assert np.all(permutation[i,j,:] == [i,j])

        
