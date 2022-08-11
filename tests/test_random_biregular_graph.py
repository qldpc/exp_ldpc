from qldpc import random_biregular_graph, remove_short_cycles
from qldpc.random_biregular_graph import _bfs_girth
import networkx as nx
import pytest

def check_biregular(G, data_degree, check_degree, check_type=True):
    if check_type is True:
        assert type(G) is nx.Graph
    # Consistency check
    for (node, degree) in G.degree():
        if G.nodes[node]['bipartite'] == 0:
            assert degree == data_degree
        else:
            assert degree == check_degree

def check_girth(G, girth_bound):
    '''Check that the girth is strictly greater than girth_bound'''
    for node in G.nodes:
        assert _bfs_girth(G, node, girth_bound) is None

seeds = [
    0x59824c5a, 0x9dca707a, 0xe0218aa8, 0x81da8035, 
    0x63b16deb, 0x7dc89245, 0x1ab46afa, 0x5cc6d93e, 
    0x6a550348, 0x97090396, 0x2a18366d, 0xcba46c36, 
    0xa7984b05, 0x82ee5a86, 0xb6cbf54b, 0xce8b63a4,
    ]

graph_cases = (
    [(27, 3, 4, s) for s in seeds]
    + [(10, 5, 6, s) for s in seeds]
    + [(21, 7, 8, s) for s in seeds]
    + [(27, 9, 10, s) for s in seeds]
    )

@pytest.mark.parametrize("left_vertices,right_deg,left_deg,seed", graph_cases)
def test_smoketest_random_biregular_graph(left_vertices, right_deg, left_deg, seed):
    right_vertices = left_vertices*left_deg//right_deg
    graph = random_biregular_graph(left_vertices, right_vertices, right_deg, left_deg, seed=seed)
    check_biregular(graph, right_deg, left_deg)

@pytest.mark.parametrize("seed", seeds)
def test_remove_short_cycles(seed):
    left_deg = 4
    right_deg = 3
    left_vertices = 51
    right_vertices = left_vertices*left_deg//right_deg
    graph = random_biregular_graph(left_vertices, right_vertices, right_deg, left_deg, seed=seed)

    remove_short_cycles(graph, 4)
    check_girth(graph, 4)

    check_biregular(graph, right_deg, left_deg)
