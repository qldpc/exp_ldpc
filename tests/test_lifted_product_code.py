from qldpc.lifted_product_code import random_abelian_generators, morgenstern_generators, _dfs_generators
from qldpc import lifted_product_code_cyclic, lifted_product_code_pgl2

def test_random_abelian_generators():
    q = 3
    m = 4
    k = 5
    generators = random_abelian_generators(q,m,k, seed=42)
    group = _dfs_generators(generators[0].identity(), generators)
    assert len(group) == q**m

def test_morgenstern_generators():
    l = 1
    i = 2
    generators = morgenstern_generators(l,i)
    identity = generators[0].identity()
    assert len(generators) == 2**l + 1
    
    group_elements = _dfs_generators(identity, generators)
    q = (2**l)**i
    assert len(group_elements) == (q-1)*q*(q+1)
    # Do DFS using the generators from the left and from the right to make sure we get the number of elements we expect
    # Check a \in A implies a^-1 \in A

def test_lifted_product_code_cyclic():
    # Parameters from Higgot and Breuckmann
    w = 14
    r = 5
    q = 22
    m = 1
    G = q**m
    code = lifted_product_code_cyclic(q=q, m=m, w=w, r=r, double_cover=True, compute_logicals=True, seed=42, check_complex=True)
    checks = code.checks
    logicals = code.logicals
    assert checks.num_qubits == (w**2 + 4*r**2)*G
    assert logicals.x.shape[0] >= checks.num_qubits - 2* (2*w*r*G)

def test_lifted_product_code_cyclic_Bw():
    # Don't use the double cover
    # Parameters from Higgot and Breuckmann
    w = 14
    r = 5
    q = 22
    m = 1
    G = q**m
    code = lifted_product_code_cyclic(q=q, m=m, w=w, r=r, double_cover=False, compute_logicals=True, seed=42, check_complex=True)
    checks = code.checks
    logicals = code.logicals
    assert checks.num_qubits == (w**2//4 + r**2)*G
    assert logicals.x.shape[0] >= checks.num_qubits - w*r*G
    
def test_lifted_product_code_pgl2_Bw():
    # The local code length is probably too short here
    # TODO: Combine with a second set of generators
    lifted_product_code_pgl2(1, 2, 5, compute_logicals=True, seed=42, check_complex=True, double_cover=False)

def test_lifted_product_code_pgl2():
    # The local code length is probably too short here
    # TODO: Combine with a second set of generators
    lifted_product_code_pgl2(1, 2, 5, compute_logicals=True, seed=42, check_complex=True, double_cover=True)
