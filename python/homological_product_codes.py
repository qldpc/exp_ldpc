import scipy.sparse as sparse
import networkx as nx
import itertools
from typing import Tuple
import numpy as np
from qecc_util import QuantumCodeChecks, num_cols, num_rows


def homological_product(partial_A : sparse.spmatrix, partial_B : sparse.spmatrix, check_complex = None) -> QuantumCodeChecks:
    '''Compute the homological product of two 2-complexes defined by their non-trivial boundary map
        Returns pair of boundary maps (partial_2, partial_1) of the total complex
    '''
    if check_complex is None:
        check_complex = False

    # D^A x I + I x D^B A_1 x B_1 -> A_0 x B_1 + A_1 x B_0
    partial_2 = sparse.vstack([
        sparse.kron(partial_A, sparse.identity(num_cols(partial_B))),
        sparse.kron(sparse.identity(num_cols(partial_A)), partial_B)
    ])
    
    partial_1 = sparse.hstack([
        sparse.kron(sparse.identity(num_rows(partial_A)), partial_B),
        sparse.kron(partial_A, sparse.identity(num_rows(partial_B)))
    ])

    if check_complex:
        assert np.all((partial_1 @ partial_2).data % 2 == 0)

    return (partial_2, partial_1)

def biregular_hpg(num_data : int, data_degree : int, check_degree : int, seed=None) -> QuantumCodeChecks:
    ''' Constructs a hypergraph product code defined by a single (data_degree, check_degree)-regular bipartite graph
        In the classical code, the check nodes represent a basis of 1-chains and the data nodes represent a basis of 0-chains.
        The boundary map from 1-chains to 0-chains gives the neighborhood of data nodes.
        In order to keep X and Z checks symmetric, we generate a random 2-complex and then take its product with the dual complex. 
    '''

    num_checks = (num_data * data_degree)//check_degree
    if num_checks*check_degree != num_data*data_degree:
        raise RuntimeError('Number of data bits incompatible with data and check degrees')

    # Generate classical Tanner graph
    tanner_graph = nx.bipartite.configuration_model(
        [data_degree for _ in range(num_data)], # Data marked with weight 0
        [check_degree for _ in range(num_checks)], # Checks marked with weight 1
        create_using=nx.Graph(),
        seed=seed,
        )

    # Consistency check
    for (node, degree) in tanner_graph.degree():
        if tanner_graph.nodes[node]['bipartite'] == 0:
            assert degree == data_degree
        else:
            assert degree == check_degree
    
    boundary_map = nx.bipartite.biadjacency_matrix(tanner_graph, row_order=[v for v in tanner_graph.nodes if tanner_graph.nodes[v]['bipartite'] == 0]).astype(int)
    # Why aren't the checks overcomplete here?
    # There's more check nodes than data nodes when we consider them as swapped
    coboundary_map = boundary_map.transpose()

    (partial_2, partial_1) = homological_product(boundary_map, coboundary_map, check_complex=True)

    (x_checks, z_checks) = (partial_2.transpose(), partial_1)
    assert x_checks.shape == z_checks.shape # If we A (x) A instead of A (x) A* we get different shapes???
    assert num_cols(x_checks) == (num_data**2 + num_checks**2)
    return (x_checks, z_checks)


def test_smoketest_biregular_hpg():
    (x_checks, z_checks) = biregular_hpg(36, 3, 4, seed=670235982)

    assert np.all((x_checks @ z_checks.transpose()).data%2 == 0)
    print(z_checks.sum(1))
    print(x_checks.sum(1))
    print(num_cols(z_checks), num_cols(z_checks)-(num_rows(z_checks) + num_rows(x_checks)))
