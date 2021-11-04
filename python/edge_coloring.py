import networkx as nx
from networkx.algorithms import bipartite
import itertools
from collections import namedtuple
import numpy as np

def edge_color_bipartite(bipartite_graph : nx.Graph):
    '''Given a bipartite graph, return an optimal edge coloring in time O(|VG||EG|).
    This uses the construction in Konz's proof that all bipartite graphs are class 1.'''

    def canonicalize_edge(e):
        (u,v) = e
        return (u,v) if u < v else (v,u)

    G = bipartite_graph.to_undirected()
    vertex_coloring = bipartite.color(G)

    if nx.number_of_selfloops(G) > 0:
        raise RuntimeError("Graph must not contain self loops")

    graph_degree = max(map(lambda x: x[1], G.degree()))

    ColorSet = namedtuple('ColorSets', ['vertices', 'edges'])
    colorings = [ColorSet(set(), set()) for _ in range(graph_degree)]

    for edge in G.edges():
        (u, v) = edge
        u_set = None
        try:
            u_set = next(x for x in colorings if u not in x.vertices and v not in x.vertices)
        except StopIteration: # There does not exist a set that does not contain u and does not contain v
            print([len(x.edges) for x in colorings])

            u_set = next(x for x in colorings if u not in x.vertices)
            v_set = next(x for x in colorings if v not in x.vertices)

            # Reform u_set and v_set s.t. we can add the edge to u_set
            # Compute an edge 2-coloring in the subgraph v_set \cup u_set
            # The new sets are the edge colors of this subgraph

            uv_subgraph = nx.subgraph_view(G,
                filter_node = lambda x: (x in u_set.vertices or x in v_set.vertices),
                filter_edge = lambda u, v: (canonicalize_edge((u,v)) in u_set.edges or canonicalize_edge((u,v)) in v_set.edges))

            # Fix edge coloring in u_set and v_set so we can add edge
            uv_component_subgraph = uv_subgraph.subgraph(nx.descendants(uv_subgraph, u))
            for uv_edge in uv_component_subgraph:
                # Each set contains disjoint edges so we remove/add the vertices accordingly 
                (remove_set, add_set) = (u_set, v_set) if canonicalize_edge(uv_edge) in u_set.edges else (v_set, u_set)

                remove_set.edges.remove(canonicalize_edge(uv_edge))
                remove_set.vertices.remove(uv_edge[0])
                remove_set.vertices.remove(uv_edge[1])

                add_set.edges.add(canonicalize_edge(uv_edge))
                add_set.vertices.add(uv_edge[0])
                add_set.vertices.add(uv_edge[1])

        # Add the original edge
        u_set.vertices.add(u)
        u_set.vertices.add(v)
        u_set.edges.add(canonicalize_edge(edge))
    
    return [v.edges for v in colorings]

def test_bipartite_edge_coloring():
    for _ in range(100):
        n_nodes = np.random.randint(10, 100)
        m_nodes = np.random.randint(10, 100)
        degree = np.random.randint(4, 20)
    
        test_graph = bipartite.generators.random_graph(n_nodes, m_nodes, degree/np.sqrt(n_nodes*m_nodes))
        print(test_graph)

        colored_sets = edge_color_bipartite(test_graph)

        for edge in test_graph.edges():
            # Each edge is colored exactly once
            assert sum(1 for coloring in colored_sets if edge in coloring) == 1

        for node in test_graph.nodes():
            # Any two edges incident to a node have different colors
            assert test_graph.degree(node) == len(set(map(lambda e: next(i for (i,c) in enumerate(colored_sets)), test_graph.edges(node))))



