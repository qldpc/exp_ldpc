from .homological_product_code import homological_product
import networkx as nx
import numpy as np
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_cols, num_rows
from .random_biregular_graph import random_biregular_graph, remove_short_cycles
from .linalg import get_rank
import warnings

def biregular_hgp(num_data : int, data_degree : int, check_degree : int, check_complex=None, seed=None, graph_multiedge_retries=None, compute_logicals=None, girth_bound=None, girth_bound_patience=None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    ''' Constructs a hypergraph product code defined by a single (data_degree, check_degree)-regular bipartite graph
        In the classical code, the check nodes represent a basis of 1-chains and the data nodes represent a basis of 0-chains.
        The boundary map from 1-chains to 0-chains gives the neighborhood of data nodes.
        In order to keep X and Z checks symmetric, we generate a random 2-complex and then take its product with the dual complex. 
        If girth_bound is not None, we remove all cycles of length up to girth_bound from the classical tanner graphs
    '''

    num_checks = (num_data * data_degree)//check_degree
    tanner_graph = random_biregular_graph(num_checks, num_data, data_degree, check_degree, seed=seed, graph_multiedge_retries=graph_multiedge_retries)
    if girth_bound is not None:
        if girth_bound_patience is None:
            girth_bound_patience = 10000
        remove_short_cycles(tanner_graph, girth_bound, seed=seed+1 if seed is not None else None, patience=girth_bound_patience)

    # Some behavior here will change with networkx 3.0
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        boundary_map = nx.bipartite.biadjacency_matrix(tanner_graph, row_order=[v for v in tanner_graph.nodes if tanner_graph.nodes[v]['bipartite'] == 0]).astype(int)
    coboundary_map = boundary_map.transpose()

    checks, logicals = homological_product(boundary_map, coboundary_map, check_complex=check_complex, compute_logicals=compute_logicals)

    assert len(logicals.x) == len(logicals.z)
    assert checks.x.shape == checks.z.shape
    assert checks.num_qubits == (num_data**2 + num_checks**2)
    return (checks, logicals)

def random_test_hgp(compute_logicals=None) -> (QuantumCodeChecks, QuantumCodeLogicals):
    if compute_logicals is None:
        compute_logicals=True
    return biregular_hgp(36, 3, 4, seed=42, compute_logicals=compute_logicals, girth_bound=4)

def test_smoketest_biregular_hgp():
    (checks, logicals) = random_test_hgp()

    # Checks commute
    assert np.all((checks.x @ checks.z.transpose()).data%2 == 0)

    # Z logicals commute with X checks
    assert np.all((checks.x @ logicals.z.transpose())%2 == 0)
    # X logicals commute with Z checks
    assert np.all((checks.z @ logicals.x.transpose())%2 == 0)

    assert get_rank(logicals.x) == logicals.x.shape[0]
    assert get_rank(logicals.z) == logicals.z.shape[0]
    # X and Z logicals come in pairs
    assert np.all(logicals.z @ logicals.x.transpose() == np.identity(logicals.z.shape[0]))

    # In general the checks may not be independent ex. toric code
    x_checks_dense = checks.x.todense()
    z_checks_dense = checks.z.todense()
    
    x_checks_rank = get_rank(x_checks_dense)
    z_checks_rank = get_rank(z_checks_dense)

    # X logicals are non-trivial
    assert get_rank(np.vstack([x_checks_dense, logicals.x])) == x_checks_rank + logicals.z.shape[0]

    # Z logicals are non-trivial
    assert get_rank(np.vstack([z_checks_dense, logicals.z])) == z_checks_rank + logicals.z.shape[0]
