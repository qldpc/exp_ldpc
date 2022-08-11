from qldpc import read_check_generators, write_check_generators
from qldpc.code_examples import d3_rotated_surface_code, random_test_hgp
from io import StringIO

def test_check_io():
    for checks in [d3_rotated_surface_code(), random_test_hgp()[0]]:
        # Write the code out
        test_buffer = StringIO()
        write_check_generators(test_buffer, checks)

        # Read it back
        test_buffer.seek(0)
        new_checks = read_check_generators(test_buffer, validate_stabilizer_code=True)

        # Should be identity
        assert (new_checks.x != checks.x).nnz == 0
        assert (new_checks.z != checks.z).nnz == 0
        assert new_checks.num_qubits == checks.num_qubits


