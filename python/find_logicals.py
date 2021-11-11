from .quantum_code_io import read_check_generators, write_check_generators
from .qecc_util import QuantumCodeChecks, QuantumCodeLogicals, num_rows, num_cols, make_check_matrix
from scipy import sparse
from typing import Iterable, Set
from sys import argv
from pathlib import Path

from sage.all import *


def matrix_to_module_elements(matrix, basis):
    return list(sum(basis[i] for (i, v) in enumerate(g) if v==1) for g in matrix)

def sparse_to_support(sparse_matrix : sparse.spmatrix):# -> Iterable[Set[int]]:
    '''Convert a scipy sparse matrix to a iter of sets indicating non-zero entries in each row'''
    return (set(col for (_, col, _) in sparse.find(sparse_matrix[row_index, :])) for row_index in range(num_rows(sparse_matrix)))

def sparse_to_f2_matrix(sparse_matrix : sparse.spmatrix):
    '''Convert a scipy sparse matrix to a dense GF(2) matrix'''
    def make_row(entries):
        return [1 if i in entries else 0 for i in range(num_cols(entries))]
    return Matrix(GF(2), map(make_row, sparse_to_support(sparse_matrix)))


def get_kernel(check_matrix):
    '''Returns a basis for the kernel of check_matrix'''
    # LinearCode constructor expected a generator matrix so we are actually specifying the dual code here
    # The check matrix of the dual code is the generator matrix of the original code
    return LinearCode(check_matrix).parity_check_matrix()

def find_logicals(checks : QuantumCodeChecks) -> QuantumCodeLogicals:
    (x_checks, z_checks, num_qubits) = checks
    
    module = CombinatorialFreeModule(GF(2), range(num_qubits), prefix="x")
    module_basis = module.basis()

    def find_logical_single_basis(checks, conjugate_checks):
        # Find the kernel of the check matrix
        kernel = get_kernel(sparse_to_f2_matrix(checks))
        
        # Reduce the kernel mod the image of the boundary operator (transpose of conjugate check matrix)
        conjugate_submodule = module.submodule(matrix_to_module_elements(conjugate_checks, module_basis))
        return make_check_matrix(set(g.support() for g in set(conjugate_submodule.reduce(g) for g in matrix_to_module_elements(kernel, module_basis)) if len(g.monomials()) > 0))

    return (find_logical_single_basis(x_checks, z_checks), find_logical_single_basis(z_checks, x_checks), num_qubits)

def test_find_logicals():
    from .code_examples import d3_rotated_surface_code
    checks = d3_rotated_surface_code()
    logicals = find_logicals(checks)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('''
            Computes the logical operators of a code given a path to the file containing the stabilizer generator specification
            ''')
    else:
        check_file_path = Path(sys.argv[1])
        with check_file_path.open('r') as check_file:
            checks = read_check_generators(check_file)
            logicals = find_logicals(checks)
            write_check_generators(sys.stdout, logicals)