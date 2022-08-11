from .homological_product_code import homological_product
import networkx as nx
from .qecc_util import QuantumCode
from .random_biregular_graph import random_biregular_graph, remove_short_cycles
import warnings

def biregular_hgp(num_data : int, data_degree : int, check_degree : int, check_complex=None, seed=None, graph_multiedge_retries=None, compute_logicals=None, girth_bound=None, girth_bound_patience=None) -> QuantumCode:
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

    code = homological_product(boundary_map, coboundary_map, check_complex=check_complex, compute_logicals=compute_logicals)
    logicals = code.logicals
    checks = code.checks
    
    assert len(logicals.x) == len(logicals.z)
    assert checks.x.shape == checks.z.shape
    assert checks.num_qubits == (num_data**2 + num_checks**2)
    return code

def random_test_hgp(compute_logicals=None) -> QuantumCode:
    if compute_logicals is None:
        compute_logicals=True
    return biregular_hgp(36, 3, 4, seed=42, compute_logicals=compute_logicals, girth_bound=4)
