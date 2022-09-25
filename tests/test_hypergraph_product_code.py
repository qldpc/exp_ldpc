from qldpc.hypergraph_product_code import random_test_hgp
import numpy as np
from qldpc.linalg import get_rank

def test_smoketest_biregular_hgp():
    code = random_test_hgp()
    checks = code.checks
    logicals = code.logicals

    # Checks commute
    assert np.all((checks.x @ checks.z.transpose()).data%2 == 0)

    # Z logicals commute with X checks
    assert np.all((checks.x @ logicals.z.transpose())%2 == 0)
    # X logicals commute with Z checks
    assert np.all((checks.z @ logicals.x.transpose())%2 == 0)

    assert get_rank(logicals.x) == logicals.x.shape[0]
    assert get_rank(logicals.z) == logicals.z.shape[0]
    # X and Z logicals come in pairs
    assert np.all((logicals.z @ logicals.x.transpose())%2 == np.identity(logicals.z.shape[0]))

    # In general the checks may not be independent ex. toric code
    x_checks_dense = checks.x.todense()
    z_checks_dense = checks.z.todense()
    
    x_checks_rank = get_rank(x_checks_dense)
    z_checks_rank = get_rank(z_checks_dense)

    # X logicals are non-trivial
    assert get_rank(np.vstack([x_checks_dense, logicals.x])) == x_checks_rank + logicals.z.shape[0]

    # Z logicals are non-trivial
    assert get_rank(np.vstack([z_checks_dense, logicals.z])) == z_checks_rank + logicals.z.shape[0]
