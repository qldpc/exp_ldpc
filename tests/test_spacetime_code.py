from qldpc import code_examples, SpacetimeCode, SpacetimeCodeSingleShot

def test_smoketest_spacetime_code():
    code = code_examples.random_test_hgp()

    SpacetimeCode(code.checks.z, 0)
    SpacetimeCode(code.checks.z, 3)

def test_smoketest_single_shot():
    code = code_examples.random_test_hgp()
    SpacetimeCodeSingleShot(code.checks.z)