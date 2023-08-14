from qldpc.qc_lifted_product_code import qc_lifted_product_code, shifts_to_polynomials
import numpy as np

def test_qc_lifted_product_code():
    # From PK'20 arxiv:2012.04068
    shift_matrix = np.array([[1,2,4,8,16],[5,10,20,9,18],[25,19,7,14,28]])
    code = qc_lifted_product_code(shifts_to_polynomials(shift_matrix), l=31, check_complex=True, compute_logicals=True)
    assert code.num_qubits == 1054
    assert code.num_logicals == 140
    
    
