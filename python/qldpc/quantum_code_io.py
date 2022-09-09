from io import IOBase
from scipy import sparse
import numpy as np
from .qecc_util import QuantumCode, QuantumCodeChecks, QuantumCodeLogicals, make_check_matrix, num_rows, num_cols

def read_quantum_code(stream : IOBase, validate_stabilizer_code = None) -> QuantumCode:
    if validate_stabilizer_code is None:
        validate_stabilizer_code = True

    lines = stream.readlines()
    # Strip out comments and whitespace
    lines = [s.split() for s in lines if s[0] != 'c']
    lines = [l for l in lines if len(l) > 0]

    if lines[0][0] != 'qecc' or len(lines[0]) != 5:
        raise RuntimeError('Invalid header. Expected qecc <# qubits> <# X checks> <# Z checks> <# logicals>')
    
    qubit_count, x_check_count, z_check_count, logical_count = int(lines[0][1]), int(lines[0][2]), int(lines[0][3]), int(lines[0][4])
    check_count = x_check_count + z_check_count

    if check_count > qubit_count:
        raise RuntimeError(f'Code overconstrained. Got {check_count} checks on {qubit_count} qubits')
    
    rows = {'X':[], 'Z':[], 'LX':[], 'LZ':[]}

    for l in lines[1:]:
        support = [int(v) for v in l[:-1]]
        check_type = l[-1]
        if check_type not in rows.keys():
            raise RuntimeError(f'Invalid check/logical type in line: \n {l}')
        for v in support:
            if v >= qubit_count:
                raise RuntimeError(f'Out of bounds check support: \n {l}')

        rows[check_type].append(support)
    
    if len(rows['X']) + len(rows['Z']) != check_count:
        raise RuntimeError(f'Number of checks does not match header. Expected {x_check_count} + {z_check_count}. Got {len(rows["X"])} + {len(rows["Z"])}')

    if len(rows['LZ']) != len(rows['LX']):
        raise RuntimeError(f'Number of X and Z logicals does not match: {len(rows["LX"])} X logicals and {len(rows["LZ"])} Z logicals')

    if len(rows['LZ']) != logical_count:
        raise RuntimeError(f'Parsed number of logicals does not match header. Expected {logical_count}. Got {len(rows["LZ"])}')
    
    x_checks = make_check_matrix(rows['X'], qubit_count)
    z_checks = make_check_matrix(rows['Z'], qubit_count)
    checks = QuantumCodeChecks(x_checks, z_checks)
    logicals = QuantumCodeLogicals(make_check_matrix(rows['LX'], qubit_count).todense(), make_check_matrix(rows['LZ'], qubit_count).todense())

    if validate_stabilizer_code is True:
        if not np.all((checks.x @ checks.z.transpose()).data%2 == 0):
            raise RuntimeError(f'X and Z checks do not generate an abelian group')

        if logicals.num_logicals > 0:
            if not np.all((checks.x @ logicals.z.transpose())%2 == 0):
                raise RuntimeError(f'Z logicals do not commute with X checks')

            if not np.all((checks.z @ logicals.x.transpose())%2 == 0):
                raise RuntimeError(f'X logicals do not commute with Z checks')

    return QuantumCode(checks, logicals)

def write_quantum_code(stream : IOBase, code : QuantumCode):
    # Header
    stream.write(f'qecc {code.num_qubits} {num_rows(code.checks.x)} {num_rows(code.checks.z)} {code.num_logicals}\n')
    # Check generators for each type
    for (entry_type, matrix) in (('X', code.checks.x), ('Z', code.checks.z), ('LZ', code.logicals.z), ('LX', code.logicals.x)):
        for row_index in range(num_rows(matrix)):
            col_list = " ".join(str(col) for col in sparse.find(matrix[row_index, :])[1])
            stream.write(f'{col_list} {entry_type}\n')
