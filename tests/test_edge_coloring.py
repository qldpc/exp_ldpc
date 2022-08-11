from qldpc import edge_color_bipartite
from qldpc.edge_coloring import _edges_with_keys, _canonicalize_edge
from networkx.algorithms import bipartite
import numpy as np

def check_graph(test_graph):
        colored_sets = edge_color_bipartite(test_graph)

        assert(len(colored_sets) == max(map(lambda x: x[1], test_graph.degree())))

        for edge in _edges_with_keys(test_graph):
            # Each edge is colored exactly once
            assert sum(1 for coloring in colored_sets if _canonicalize_edge(edge) in coloring) == 1
        for node in test_graph.nodes():
            # The number of unique colors incident to a vertex is equal to the degree
            adjacent_colors = list(map(lambda e: next(i for (i,c) in enumerate(colored_sets) if _canonicalize_edge(e) in c), test_graph.edges(node)))
            assert len(adjacent_colors) == len(set(adjacent_colors))


def test_bipartite_edge_coloring():
    for _ in range(100):
        n_nodes = np.random.randint(10, 100)
        m_nodes = np.random.randint(10, 100)
        degree = np.random.randint(4, 20)
    
        test_graph = bipartite.generators.random_graph(n_nodes, m_nodes, degree/np.sqrt(n_nodes*m_nodes))
        check_graph(test_graph)

def test_bipartite_multigraph_edge_coloring():
    for _ in range(100):
        n_nodes = np.random.randint(10, 100)
        m_nodes = np.random.randint(10, 100)

        n_degree_sequence = np.random.randint(4, 20, size=n_nodes)
        avg_m_edges = np.sum(n_degree_sequence)/m_nodes
        m_degree_sequence = np.random.randint(int(np.maximum(0,avg_m_edges-4)), int(avg_m_edges+4), size=m_nodes)

        # Dump all the excess edges on the last node
        m_excess = np.sum(m_degree_sequence) - np.sum(n_degree_sequence)
        if m_excess > 0:
            n_degree_sequence[-1] += m_excess
        else:
            m_degree_sequence[-1] -= m_excess
    
        test_graph = bipartite.generators.configuration_model(n_degree_sequence, m_degree_sequence)
        check_graph(test_graph)
    