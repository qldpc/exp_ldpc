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
    
    
