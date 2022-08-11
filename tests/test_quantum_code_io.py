from qldpc import read_check_generators, write_check_generators
from qldpc.code_examples import random_test_hgp
from io import StringIO

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


