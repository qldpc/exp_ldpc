from qldpc import read_check_generators, write_check_generators, read_logicals, write_logicals
from qldpc.code_examples import random_test_hgp
from io import StringIO
import numpy as np

def test_check_io():
    for code in [random_test_hgp()]:
        # Write the code out
        test_buffer = StringIO()
        write_check_generators(test_buffer, code.checks)

        # Read it back
        test_buffer.seek(0)
        new_checks = read_check_generators(test_buffer, validate_stabilizer_code=True)

        # Should be identity
        assert (new_checks.x != code.checks.x).nnz == 0
        assert (new_checks.z != code.checks.z).nnz == 0
        assert new_checks.num_qubits == code.checks.num_qubits

def test_check_io_logicals():
    for code in [random_test_hgp()]:
        # Write the code out
        test_buffer = StringIO()
        write_logicals(test_buffer, code.logicals)

        # Read it back
        test_buffer.seek(0)
        new_logicals = read_logicals(test_buffer)

        # Should be identity
        assert np.all(new_logicals.x == code.logicals.x)
        assert np.all(new_logicals.z == code.logicals.z)
        assert new_logicals.num_qubits == code.logicals.num_qubits
