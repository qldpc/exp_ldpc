from qldpc import spacetime_code
from qldpc import code_examples

def test_smoketest_spacetime_code():
    code = code_examples.random_test_hgp()

    spacetime_code.spacetime_code(code.checks.z, 0)
    spacetime_code.spacetime_code(code.checks.z, 3)

def test_smoketest_single_shot():
    code = code_examples.random_test_hgp()
    spacetime_code.single_shot(code.checks.z)