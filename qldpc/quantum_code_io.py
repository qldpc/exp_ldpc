from io import IOBase
import scipy
from scipy import sparse
import numpy as np
from .qecc_util import QuantumCodeChecks, make_check_matrix, num_rows, num_cols

def read_check_generators(stream : IOBase, validate_stabilizer_code = None) -> QuantumCodeChecks:
    if validate_stabilizer_code is None:
        validate_stabilizer_code = True

    lines = stream.readlines()
    # Strip out comments and whitespace
    lines = [s.split() for s in lines if s[0] != 'c']
    lines = [l for l in lines if len(l) > 0]

    if lines[0][0] != 'qecc' or len(lines[0]) != 4:
        raise RuntimeError('Invalid header. Expected qecc <# qubits> <# X checks> <# Z checks>')
    
    qubit_count, x_check_count, z_check_count = int(lines[0][1]), int(lines[0][2]), int(lines[0][3])
    check_count = x_check_count + z_check_count

    if check_count > qubit_count:
        raise RuntimeError(f'Code overconstrained. Got {check_count} checks on {qubit_count} qubits')

    x_checks = []
    z_checks = []

    for l in lines[1:]:
        support = [int(v) for v in l[:-1]]
        check_type = l[-1]
        if check_type != 'X' and check_type != 'Z':
            raise RuntimeError(f'Invalid check type in line: \n {l}')
        for v in support:
            if v >= qubit_count:
                raise RuntimeError(f'Out of bounds check support: \n {l}')

        if check_type == 'X':
            x_checks.append(support)
        else:
            z_checks.append(support)
        
    if len(z_checks) + len(x_checks) != check_count:
        raise RuntimeError(f'Number of checks does not match parsed number of lines')

    if len(z_checks) != len(x_checks):
        raise RuntimeError(f' Number of X checks does not match number of Z checks, got {x_checks} and {z_checks} respectively.')
    
    x_checks = make_check_matrix(x_checks, qubit_count)
    z_checks = make_check_matrix(z_checks, qubit_count)

    if validate_stabilizer_code is True:
        if not np.all((x_checks @ z_checks.transpose()).data%2 == 0):
            raise RuntimeError(f'X and Z checks do not generate an abelian group')

    return (x_checks, z_checks, qubit_count)

def write_check_generators(stream : IOBase, checks : QuantumCodeChecks):
    (x_checks, z_checks, num_qubits) = checks
    
    assert num_cols(x_checks) == num_cols(z_checks)
    assert num_cols(x_checks) == num_qubits
    # Header
    stream.write(f'qecc {num_qubits} {num_rows(x_checks)} {num_rows(z_checks)}\n')
    # Check generators for each type
    for (check_type, check_matrix) in (('X', x_checks), ('Z', z_checks)):
        for row_index in range(num_rows(check_matrix)):
            col_list = " ".join(str(col) for col in sparse.find(check_matrix[row_index, :])[1])
            stream.write(f'{col_list} {check_type}\n')

def test_check_io():
    from .code_examples import d3_rotated_surface_code, random_test_hpg
    from io import StringIO

    for checks in [d3_rotated_surface_code(), random_test_hpg()[0]]:
        # Write the code out
        test_buffer = StringIO()
        write_check_generators(test_buffer, checks)

        # Read it back
        test_buffer.seek(0)
        new_checks = read_check_generators(test_buffer, validate_stabilizer_code=True)

        # Should be identity
        assert (new_checks[0] != checks[0]).nnz == 0
        assert (new_checks[1] != checks[1]).nnz == 0
        assert new_checks[2] == checks[2]

