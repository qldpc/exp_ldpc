from .homological_product_code import homological_product
import networkx as nx
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows
from .random_biregular_graph import random_biregular_graph
from .generator_matrix import get_rank

def biregular_hpg(num_data : int, data_degree : int, check_degree : int, seed=None, graph_multiedge_retries=None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    ''' Constructs a hypergraph product code defined by a single (data_degree, check_degree)-regular bipartite graph
        In the classical code, the check nodes represent a basis of 1-chains and the data nodes represent a basis of 0-chains.
        The boundary map from 1-chains to 0-chains gives the neighborhood of data nodes.
        In order to keep X and Z checks symmetric, we generate a random 2-complex and then take its product with the dual complex. 
    '''

    num_checks = (num_data * data_degree)//check_degree
    tanner_graph = random_biregular_graph(num_checks, num_data, data_degree, check_degree, seed=seed, graph_multiedge_retries=graph_multiedge_retries)
    
    boundary_map = nx.bipartite.biadjacency_matrix(tanner_graph, row_order=[v for v in tanner_graph.nodes if tanner_graph.nodes[v]['bipartite'] == 0]).astype(int)
    coboundary_map = boundary_map.transpose()

    ((partial_2, partial_1, num_qubits), (x_logicals, z_logicals, _)) = homological_product(boundary_map, coboundary_map, check_complex=True)

    (x_checks, z_checks) = (partial_2.transpose().tocsr(), partial_1.tocsr())

    assert len(x_logicals) == len(z_logicals)
    assert x_checks.shape == z_checks.shape
    assert num_qubits == (num_data**2 + num_checks**2)
    return ((x_checks, z_checks, num_qubits), (x_logicals, z_logicals, num_qubits))

def random_test_hpg() -> (QuantumCodeChecks, QuantumCodeLogicals):
    return biregular_hpg(36, 3, 4, seed=42)

def test_smoketest_biregular_hpg():
    ((x_checks, z_checks, _), (x_logicals, z_logicals, _)) = random_test_hpg()

    z_logicals = np.vstack(z_logicals)
    x_logicals = np.vstack(x_logicals)
    
    # Checks commute
    assert np.all((x_checks @ z_checks.transpose()).data%2 == 0)

    # Z logicals commute with X checks
    assert np.all((x_checks @ z_logicals.transpose())%2 == 0)
    # X logicals commute with Z checks
    assert np.all((z_checks @ x_logicals.transpose())%2 == 0)

    # X and Z logicals come in pairs
    print((z_logicals @ x_logicals.transpose())%2)
    assert np.count_nonzero((z_logicals @ x_logicals.transpose())%2) == z_logicals.shape[0]
    b
    # In general the checks may not be independent ex. toric code
    x_checks = x_checks.todense()
    z_checks = z_checks.todense()
    
    x_checks_rank = get_rank(x_checks)
    z_checks_rank = get_rank(z_checks)

    # X logicals are non-trivial
    assert get_rank(np.vstack([x_checks, x_logicals])) == x_checks_rank + x_logicals.shape[0]

    # Z logicals are non-trivial
    assert get_rank(np.vstack([z_checks, z_logicals])) == z_checks_rank + z_logicals.shape[0]
    
    print(f'Z check degree: {z_checks.sum(1)}')
    print(f'X check degree: {x_checks.sum(1)}')
    print(num_cols(z_checks), num_cols(z_checks)-(num_rows(z_checks) + num_rows(x_checks)))
