import networkx as nx
import numpy as np
import itertools
from collections import deque
from typing import Tuple, Optional
from scipy.stats import rv_discrete

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

def _search_cycle(graph, source, depth_limit : int) -> Optional[Tuple[int, Tuple[int, int]]]:
    '''Returns (l, edge) where edge is contained in a length l cycle through source.
    When the graph is bipartite, this edge is contained in a shortest cycle.
    Otherwise, the edge is contained in a cycle of length at most 1 greater than the shortest cycle.
    Searches only up to depth_limit. For bipartite graphs a cycle will only be found if its length is <= 2*search_depth
    I. Alon and M. Rodeh, SIAM J. Comput. 7(4), 413-423 (1978).
    '''

    level = dict()
    traverse = deque()
    traverse.append(source)
    level[source] = 0

    while len(traverse) > 0:
        u = traverse.popleft()
        u_level = level[u]

        for neighbor in graph.neighbors(u):
            n_level = level.get(neighbor, None)
            if n_level is None:
                n_level = u_level+1
                level[neighbor] = n_level
                if n_level < depth_limit:
                    traverse.append(neighbor)
            else:
                if u_level <= n_level:
                    return (2*(u_level+1), (u, neighbor))
    return None


def remove_short_cycles(graph : nx.Graph, girth_bound : int, seed=None, patience=1000000):
    '''Remove short cycles in a bipartite graph (inplace) by swapping edges at random so that the girth is strictly greater than girth_bound'''

    depth_limit = girth_bound//2
    left_vertex_set = np.array(nx.subgraph_view(graph, filter_node=lambda node: graph.nodes[node]['bipartite'] == 0).nodes())
    rng = np.random.default_rng(seed=seed)
    # Amortize the exit check between this many random vertex selections
    exit_check_interval = left_vertex_set.shape[0]*10

    # Networkx does not expose a method to sample edges in O(1) without ammortizing an O(n) operation that is invalidated after mutating the graph
    # So we need to do this manually
    # Sample a vertex weighted by its degree and then sample a random edge
    # Equivalent to sampling edges at random
    vertices = np.fromiter(graph.nodes(), dtype=np.int32)
    degrees = np.fromiter(map(lambda n: graph.degree(n), vertices), dtype=np.int32)
    vertex_distr = rv_discrete(values=(vertices, degrees/np.sum(degrees)))

    # Check if all short cycles have been removed
    def full_clear():
        return all(map(lambda v: _search_cycle(graph, v, depth_limit=depth_limit) is None, left_vertex_set))

    for t in range(patience):
        # Exit condition
        if t%exit_check_interval == 0 and full_clear():
            break

        # Select a node at random to check the girth condition
        node = rng.choice(left_vertex_set)
        cycle = _search_cycle(graph, node, depth_limit=depth_limit)
        if cycle is not None:
            # Short cycle was found
            _, edge = cycle
            (u1, v1) = _canonicalize_edge_bipartite(graph, edge)

            # Attempt to swap it with another edge
            for _ in range(patience):
                # Select new edge
                rand_vertex = vertex_distr.rvs(random_state = rng)
                candidate_edge = rng.choice(list(graph.edges(rand_vertex)))
                # Ensure we can swap it without making a multi-edge
                (u2, v2) = _canonicalize_edge_bipartite(graph, candidate_edge[:2])
                # Check u2 not neighbor of v1
                # Check u1 not neighbor of v2
                neighborhood_good = u2 not in graph.neighbors(v1) and u1 not in graph.neighbors(v2)
                # Check that we do not have something like \/ formed from 3 vertices
                unique_endpoints = u1 != u2 and v1 != v2
                if unique_endpoints and neighborhood_good:
                    # Swap edges
                    graph.remove_edge(u1, v1)
                    graph.remove_edge(u2, v2)
                    graph.add_edge(u1, v2)
                    graph.add_edge(u2, v1)
                    break
            else:
                raise RuntimeError("Patience exceeded while selecting an edge to swap in short cycle removal.")
    else:
        if not full_clear():
            raise RuntimeError("Patience exceeded while removing short cycles.")