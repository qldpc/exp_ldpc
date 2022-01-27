import networkx as nx
from networkx.algorithms import bipartite
from collections import namedtuple
from typing import List, Set
import numpy as np

def _canonicalize_edge(e):
    (u,v) = e[:2] # [:2] in case the graph is weighted
    return (u,v) if u < v else (v,u)

def edge_color_bipartite(bipartite_graph : nx.Graph) -> List[Set[int]]:
    '''Given a bipartite graph, return an optimal edge coloring in time O(|VG||EG|).
    This uses the construction in Konz's proof that all bipartite graphs are class 1.'''

    G = bipartite_graph.to_undirected()

    if not nx.is_bipartite(G):
        raise RuntimeError("Graph must be bipartite")

    if nx.number_of_selfloops(G) > 0:
        raise RuntimeError("Graph must not contain self loops")

    graph_degree = max(map(lambda x: x[1], G.degree()))

    ColorSet = namedtuple('ColorSets', ['vertices', 'edges'])
    colorings = [ColorSet(set(), set()) for _ in range(graph_degree)]

    for edge in G.edges():
        (u, v) = edge[:2]
        u_set = None
        try:
            u_set = next(x for x in colorings if u not in x.vertices and v not in x.vertices)
        except StopIteration: # There does not exist a set that does not contain u and does not contain v
            u_set = next(x for x in colorings if u not in x.vertices)
            v_set = next(x for x in colorings if v not in x.vertices)

            # Reform u_set and v_set s.t. we can add the edge to u_set
            # Compute an edge 2-coloring in the subgraph v_set \cup u_set
            # The new sets are the edge colors of this subgraph

            uv_subgraph = nx.subgraph_view(G,
                filter_node = lambda x: (x in u_set.vertices or x in v_set.vertices),
                filter_edge = lambda u, v: (_canonicalize_edge((u,v)) in u_set.edges or _canonicalize_edge((u,v)) in v_set.edges))

            # Fix edge coloring in u_set and v_set so we can add edge
            # We do this by following the chain of edges incident to v and swapping all the colors in this chain
            # ==v_set== u      v ==u_set== x ==v_set== x ==u_set== ....

            u_to_v_set = set()
            v_to_u_set = set()
            for uv_edge in nx.edge_dfs(uv_subgraph, v):
                if _canonicalize_edge(uv_edge) in u_set.edges:
                    u_to_v_set.add(_canonicalize_edge(uv_edge))
                else:
                    v_to_u_set.add(_canonicalize_edge(uv_edge))

            u_to_v_vertices = set(v for edges in u_to_v_set for v in edges)
            v_to_u_vertices = set(v for edges in v_to_u_set for v in edges)

            u_set.edges.difference_update(u_to_v_set)
            u_set.vertices.difference_update(u_to_v_vertices)
            u_set.edges.update(v_to_u_set)
            u_set.vertices.update(v_to_u_vertices)

            v_set.edges.difference_update(v_to_u_set)
            v_set.vertices.difference_update(v_to_u_vertices)
            v_set.edges.update(u_to_v_set)
            v_set.vertices.update(u_to_v_vertices)

        # Add the original edge
        u_set.vertices.add(u)
        u_set.vertices.add(v)
        u_set.edges.add(_canonicalize_edge(edge))
    
    return [v.edges for v in colorings]

def test_bipartite_edge_coloring():
    for _ in range(100):
        n_nodes = np.random.randint(10, 100)
        m_nodes = np.random.randint(10, 100)
        degree = np.random.randint(4, 20)
    
        test_graph = bipartite.generators.random_graph(n_nodes, m_nodes, degree/np.sqrt(n_nodes*m_nodes))
        print(test_graph)

        colored_sets = edge_color_bipartite(test_graph)

        assert(len(colored_sets) == max(map(lambda x: x[1], test_graph.degree())))

        for edge in test_graph.edges():
            # Each edge is colored exactly once
            assert sum(1 for coloring in colored_sets if _canonicalize_edge(edge) in coloring) == 1
        for node in test_graph.nodes():
            # The number of unique colors incident to a vertex is equal to the degree
            adjacent_colors = list(map(lambda e: next(i for (i,c) in enumerate(colored_sets) if _canonicalize_edge(e) in c), test_graph.edges(node)))
            assert len(adjacent_colors) == len(set(adjacent_colors))



