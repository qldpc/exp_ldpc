import warnings
from pathlib import Path
import sys
import argparse

import networkx as nx
import numpy as np
from scipy import sparse 

from qldpc import write_quantum_code, QuantumCode,QuantumCodeChecks,QuantumCodeLogicals, GF2
from qldpc.random_biregular_graph import random_biregular_graph, remove_short_cycles
from qldpc.homological_product_code import get_logicals


def random_regular_ldpc_code(num_data : int, data_degree : int, check_degree : int, seed=None, graph_multiedge_retries=None, compute_logicals=None, girth_bound=None, girth_bound_patience=None) -> QuantumCode:
    # Generate random biregular graph
    num_checks = (num_data * data_degree)//check_degree
    tanner_graph = random_biregular_graph(num_checks, num_data, data_degree, check_degree, seed=seed, graph_multiedge_retries=graph_multiedge_retries)

    # Swap edges to achieve desired girth
    if girth_bound is not None:
        if girth_bound_patience is None:
            girth_bound_patience = 10000
        remove_short_cycles(tanner_graph, girth_bound, seed=seed+1 if seed is not None else None, patience=girth_bound_patience)

    # Check matrix is biadjacency matrix
    # Some behavior here will change with networkx 3.0
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        z_checks = nx.bipartite.biadjacency_matrix(tanner_graph, row_order=[v for v in tanner_graph.nodes if tanner_graph.nodes[v]['bipartite'] == 0]).astype(int)

    # Create a code object out of it
    x_checks = np.zeros((0,z_checks.shape[1]), dtype=np.uint32)
    checks = QuantumCodeChecks(sparse.csr_matrix(x_checks), sparse.csr_matrix(z_checks))
    logicals = get_logicals(checks, True, False)
    
    code = QuantumCode(checks, logicals)
    return code

if __name__ == '__main__':

    # Arguments and such
    parser = argparse.ArgumentParser(description='Generate a classical LDPC code')
    parser.add_argument('dc', type=int, help='Check vertex degree in Tanner graph')
    parser.add_argument('dv', type=int, help='Data vertex degree in Tanner graph')
    parser.add_argument('nv', type=int, help='''
        Classical code data vertex count.''')
    parser.add_argument('--girth_bound', type=int, help='Remove all cycles of length <= girth_bound in the tanner graph', default=None)
    parser.add_argument('--girth_bound_patience', type=int, help='Timeout for random swaps to achieve girth bound', default=100000)
    parser.add_argument('--rounds', type=int, help='Number of rounds of syndrome extraction', default=1)
    parser.add_argument('--seed', type=lambda x: int(x) if x is not None else None, help='PRNG seed', default=None)
    parser.add_argument('--save_code', type=Path, help='File path to save the code to')
    parser.add_argument('--compute_logicals', type=bool, default=True, help='Optionally compute logical operators of the code, Warning: This has O(n^3) time complexity')

    args = parser.parse_args(sys.argv[1:])

    if args.save_code is not None and args.save_code.exists():
        print('Code save destination already exists')
        exit(-1)

    # Generate thes code

    code = random_regular_ldpc_code(args.nv, args.dv, args.dc, seed=args.seed, compute_logicals=args.compute_logicals,
        girth_bound=args.girth_bound, girth_bound_patience=args.girth_bound_patience)

    # Dump it
    if args.save_code is not None:
        with args.save_code.open('w') as code_file:
            write_quantum_code(code_file, code)
    else:
        write_quantum_code(sys.stdout, code)
