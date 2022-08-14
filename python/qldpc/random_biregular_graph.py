import networkx as nx
import numpy as np
import itertools
from typing import Tuple

def _canonicalize_edge(x : Tuple[int, int]) -> Tuple[int, int]:
    return (x[0], x[1]) if x[0] < x[1] else (x[1], x[0])

def _canonicalize_edge_bipartite(graph : nx.Graph, x : Tuple[int, int]) -> Tuple[int, int]:
    return (x[0], x[1]) if graph.nodes[x[1]]['bipartite'] == 1 else (x[1], x[0])

def random_biregular_graph(num_checks : int, num_data : int, data_degree : int, check_degree : int, seed=None, graph_multiedge_retries=None) -> nx.Graph:
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
            new_edge_a = _canonicalize_edge((edge_a[0], edge_b[1]))
            new_edge_b = _canonicalize_edge((edge_b[0], edge_a[1]))

            # We can end up selecting an edge that's already in the multiedge_list
            no_double_remove = _canonicalize_edge(edge_b) not in edge_removal_list
            if no_double_remove:

                edge_removal_list.append(_canonicalize_edge(edge_a))
                edge_removal_list.append(_canonicalize_edge(edge_b))

                edge_add_list.append(new_edge_a)
                edge_add_list.append(new_edge_b)
        
        # Apply update
        for e in edge_removal_list:
            # Removes an arbitrary edge if there are multiple edges like e
            tanner_graph.remove_edge(*e, key=None)
        tanner_graph.add_edges_from(edge_add_list)        
    else:
        raise RuntimeError('Unable to remove multiedges from the graph')

    tanner_graph = nx.Graph(tanner_graph)

    return tanner_graph

def _bfs_girth(graph, node, max_depth : int):
    '''Find a cycle of length up to max depth going through node via BFS'''
    try:
        return next(nx.all_simple_paths(graph, node, node, cutoff=max_depth))
    except StopIteration:
        return None

def remove_short_cycles(graph : nx.Graph, girth_bound : int, seed=None, patience=1000000):
    '''Remove short cycles in a bipartite graph (inplace) by swapping edges at random so that the girth is strictly greater than girth_bound'''
    left_vertex_set = list(nx.subgraph_view(graph, filter_node=lambda node: graph.nodes[node]['bipartite'] == 0).nodes())
    rng = np.random.default_rng(seed=seed)
    num_edges = len(graph.edges)

    for _ in range(patience):
        for node in left_vertex_set:
            path = _bfs_girth(graph, node, max_depth=girth_bound)
            if path is None:
                continue

            (u1, v1) = _canonicalize_edge_bipartite(graph, path[2:])
            # Select new edge
            for _ in range(patience):
                candidate_edge = graph.edges[rng.integers(num_edges)]
                # Ensure we can swap while maintaining a graph
                (u2, v2) = _canonicalize_edge_bipartite(candidate_edge[:2])
                # Check u2 not neighbor of v1
                # Check u1 not neighbor of v2
                if u2 not in graph.neighbors(v1) and u1 not in graph.neighbors(v2):
                    # Swap edges
                    graph.remove_edge(u1, v1)
                    graph.remove_edge(u2, v2)
                    graph.add_edge(u1, v2)
                    graph.add_edge(u2, v1)
            else:
                raise RuntimeError("Patience exceeded while removing short cycles.")