from qldpc import lifted_product_code as lp
from qldpc import matrix_lifted_product_code as mlp
from qldpc import GF2
import numpy as np

def test_matrix_lifted_product_code_Z31():
    # From PK'20 arxiv:2012.04068
    Z31 = [lp.Zqm(31,1,np.array([a],dtype=int)) for a in range(31)]
    shifts = [[1,2,4,8,16],[5,10,20,9,18],[25,19,7,14,28]]
    base_matrix = np.array([[mlp.group_algebra_monomial(GF2(1), Z31[a]) for a in row] for row in shifts])
    code = mlp.matrix_lifted_product_code(Z31, base_matrix, check_complex=True, compute_logicals=True)
    assert code.num_qubits == 1054
    assert code.num_logicals == 140
    
    
def test_matrix_lifted_product_code_B3():
    # From PK'19 arXiv:1904.02703
    # B3
    Z127 = [lp.Zqm(127,1,np.array([a],dtype=int)) for a in range(127)]

    def monomial(g):
        return mlp.group_algebra_monomial(GF2(1),g)
    shift = lambda i: monomial(Z127[i])
    zero = shift(0)*0
    one = shift(1)

    base_matrix_A = np.array(
        [[shift(  0),       zero, shift( 51), shift( 52),       zero],
         [      zero, shift(  0),       zero, shift(111), shift( 20)],
         [shift(  0),       zero, shift( 98),       zero, shift(122)],
         [shift(  0), shift( 80),       zero, shift(119),       zero],
         [      zero, shift(  0), shift(  5),       zero, shift(106)],])


    # Paper shows B should be proportional to 5x5 identity, but counting shows that B should be 1x1
    Bscale = shift(0) + shift(1) + shift(7)
    base_matrix_B = np.vectorize(lambda x: Bscale*x)(GF2.Identity(1))
    
    code = mlp.matrix_lifted_product_code(Z127, base_matrix_A, base_matrix_B, check_complex=True, compute_logicals=True)
    assert code.num_qubits == 1270
    assert code.num_logicals == 28

def test_psl_lift():
    group = list(lp.get_psl2(5))
    
    
def test_regular_repp():
    group = lp.get_psl2(5)
    rep = mlp.RegularRep(group)
    table = {}
    
    # Populate
    for g in group:
        m = rep.get_rep(g)
        assert np.all(np.count_nonzero(m==GF2(1),axis=0))
        assert np.all(np.count_nonzero(m==GF2(1),axis=1))
        table[g] = m
    # Check multiplication table
    for g in group:
        for h in group:
            p = g@h
            assert np.all(table[p] == table[g]@table[h])
        
