import networkx as nx
import numpy as np
import itertools
import pytest
from typing import Tuple

def canonicalize_edge(x : Tuple[int, int]) -> Tuple[int, int]:
    return (x[0], x[1]) if x[0] < x[1] else (x[1], x[0])

def random_biregular_graph(num_checks : int, num_data : int, data_degree : int, check_degree : int, seed=None, graph_multiedge_retries=None):
    if graph_multiedge_retries is None:
        graph_multiedge_retries = 100

    if num_checks*check_degree != num_data*data_degree:
        raise RuntimeError('Number of data bits incompatible with data and check degrees')

    # Generate classical Tanner graph
    tanner_graph = nx.bipartite.configuration_model(
        [data_degree for _ in range(num_data)], # Data marked with weight 0
        [check_degree for _ in range(num_checks)], # Checks marked with weight 1
        seed=seed,
        create_using=nx.MultiGraph(),
        )

    # The generator procedure uses the configuration model so we may have a small fraction of multiedges
    # We can randomly swap edges to get rid of them

    # Swap endpoints of edges s.t. we have no multiedges
    # Swapping ensures that we still have a biregular graph at the end
    # It is possible that this procedure creates multiedges because we do not enforce that (a, b') does not already exist

    #  a  --- b          a   b
    #                     \ / 
    #             |->      X    
    #                     / \ 
    #  a' --- b'         a'  b'

    rng = np.random.default_rng(seed=seed)
    data_vertices = [v for v, d in tanner_graph.nodes(data=True) if d['bipartite'] == 0]
    for _ in range(graph_multiedge_retries):
        multiedge_list = []
        # Create list of multiedges
        for node in data_vertices:
            for neighbor_node in tanner_graph.neighbors(node):
                num_edges = tanner_graph.number_of_edges(node, neighbor_node)
                # (node, neighbor_node) has a multiedge
                if num_edges > 1:
                    multiedge_list.extend(itertools.repeat((node, neighbor_node), num_edges-1))
    
        if len(multiedge_list) == 0:
            break
        
        # ---------------

        edge_removal_list = list()
        edge_add_list = list()

        # Compute update
        edge_list = list(tanner_graph.edges())
        swap_edges = rng.choice(edge_list, size=len(multiedge_list), replace=False)
        for (edge_a, edge_b) in zip(multiedge_list, swap_edges):
            new_edge_a = canonicalize_edge((edge_a[0], edge_b[1]))
            new_edge_b = canonicalize_edge((edge_b[0], edge_a[1]))

            # We can end up selecting an edge that's already in the multiedge_list
            if canonicalize_edge(edge_b) not in edge_removal_list:

                edge_removal_list.append(canonicalize_edge(edge_a))
                edge_removal_list.append(canonicalize_edge(edge_b))

                edge_add_list.append(new_edge_a)
                edge_add_list.append(new_edge_b)
        
        # Apply update
        for e in edge_removal_list:
            # Removes an arbitrary edge if there are multiple edges like e
            tanner_graph.remove_edge(*e, key=None)
        tanner_graph.add_edges_from(edge_add_list)
        
        check_biregular(tanner_graph, data_degree, check_degree, False)
    else:
        raise RuntimeError('Unable to remove multiedges from the graph')

    tanner_graph = nx.Graph(tanner_graph)

    return tanner_graph
        
def check_biregular(G, data_degree, check_degree, check_type=True):
    if check_type is True:
        assert type(G) is nx.Graph
    # Consistency check
    for (node, degree) in G.degree():
        if G.nodes[node]['bipartite'] == 0:
            assert degree == data_degree
        else:
            assert degree == check_degree


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
