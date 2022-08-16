from qldpc import code_examples, SpacetimeCode, SpacetimeCodeSingleShot
import numpy as np

def test_smoketest_spacetime_code():
    code = code_examples.random_test_hgp()

    SpacetimeCode(code.checks.z, 0)
    SpacetimeCode(code.checks.z, 3)

def test_smoketest_single_shot():
    code = code_examples.random_test_hgp()
    SpacetimeCodeSingleShot(code.checks.z)

def test_smoketest_spacetime_code_syndrome():
    code = code_examples.random_test_hgp()
    rounds = 3

    spacetime_code = SpacetimeCode(code.checks.z, rounds)

    dummy_history = lambda _: np.zeros(code.checks.z.shape[0])
    dummy_readout = np.zeros(code.checks.z.shape[1])

    spacetime_syndrome = spacetime_code.syndrome_from_history(dummy_history, dummy_readout)
    assert spacetime_syndrome.shape[0] == spacetime_code.spacetime_check_matrix.shape[0]