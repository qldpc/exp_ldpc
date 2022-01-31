from qLDPC.quantum_code_io import read_check_generators, write_check_generators
from qLDPC.qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_rows, num_cols, make_check_matrix
from scipy import sparse
import numpy as np
from typing import Iterable, Set
import sys
from sys import argv
from pathlib import Path

from sage.all import *

def sparse_to_support(sparse_matrix : sparse.spmatrix):# -> Iterable[Set[int]]:
    '''Convert a scipy sparse matrix to a iter of sets indicating non-zero entries in each row'''
    return (frozenset(sparse.find(sparse_matrix[row_index, :])[1]) for row_index in range(num_rows(sparse_matrix)))

def sparse_to_f2_matrix(sparse_matrix : sparse.spmatrix):
    '''Convert a scipy sparse matrix to a dense GF(2) matrix'''
    def make_row(entries):
        return [1 if i in entries else 0 for i in range(num_cols(sparse_matrix))]
    return Matrix(GF(2), map(make_row, sparse_to_support(sparse_matrix)))

def find_logicals(checks : QuantumCodeChecks) -> QuantumCodeLogicals:
    (x_checks, z_checks, num_qubits) = checks
    
    boundary_1 = sparse_to_f2_matrix(x_checks)
    boundary_2 = sparse_to_f2_matrix(z_checks).transpose()

    code_complex = ChainComplex(data={2:boundary_2, 1:boundary_1}, base_ring=GF(2), degree_of_differential=-1)
    homology_kwargs = {'deg':1, 'algorithm':'dhsw', 'generators':True, 'verbose':False}

    code_complex_homology = code_complex.homology(**homology_kwargs)
    x_logicals = np.array([v[1].vector(1) for v in code_complex_homology])

    dual_code_complex_homology = code_complex.dual().homology(**homology_kwargs)
    z_logicals = np.array([v[1].vector(1) for v in dual_code_complex_homology])

    return (x_logicals, z_logicals, num_qubits)

def test_d3_rotated_surface_code():
    from qLDPC.code_examples import d3_rotated_surface_code
    checks = d3_rotated_surface_code()
    logicals = find_logicals(checks)
    assert list(sparse_to_support(logicals[0])) == [{2, 5, 8}]
    assert list(sparse_to_support(logicals[1])) == [{6, 7, 8}]

def test_hypergraph_product_code():
    from qLDPC.code_examples import random_test_hpg
    checks = random_test_hpg()
    logicals = find_logicals(checks)
    # Does the number of logicals we found match the numbers of logicals we're supposed to have?
    assert checks[2] == 2*num_rows(checks[0]) + num_rows(logicals[0])


if __name__ == '__main__':
    sys.setrecursionlimit(3000)

    if len(sys.argv) <= 1:
        print('''
            Computes the logical operators of a code given a path to the file containing the stabilizer generator specification
            ''')
    else:
        check_file_path = Path(sys.argv[1])
        with check_file_path.open('r') as check_file:
            checks = read_check_generators(check_file, validate_stabilizer_code=True)
            logicals = find_logicals(checks)
            write_check_generators(sys.stdout, logicals)