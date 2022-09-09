from qldpc import read_quantum_code, write_quantum_code
from qldpc.code_examples import random_test_hgp
from io import StringIO
import numpy as np

def test_check_io():
    for code in [random_test_hgp()]:
        # Write the code out
        test_buffer = StringIO()
        write_quantum_code(test_buffer, code)

        # Read it back
        test_buffer.seek(0)
        new_code = read_quantum_code(test_buffer, validate_stabilizer_code=True)

        # Should be identity
        assert (new_code.checks.x != code.checks.x).nnz == 0
        assert (new_code.checks.z != code.checks.z).nnz == 0
        assert new_code.num_qubits == code.checks.num_qubits

        assert np.all(new_code.logicals.x == code.logicals.x)
        assert np.all(new_code.logicals.z == code.logicals.z)
