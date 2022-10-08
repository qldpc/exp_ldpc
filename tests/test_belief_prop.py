from qldpc import BeliefPropagation
import numpy as np
import scipy.sparse as sparse

def test_belief_prop():
    hamming_code = np.array([[1,1,0,1,1,0,0],[1,0,1,1,0,1,0],[0,1,1,1,0,0,1]], dtype=np.uint32)
    hamming_code_sparse = sparse.csr_matrix(hamming_code)

    bp = BeliefPropagation(hamming_code_sparse)

    for i in range(hamming_code.shape[1]):
        x = np.zeros(hamming_code.shape[1], dtype=np.uint32)
        x[i] = 1
        syndrome = (hamming_code @ x)%2
        llr = -np.ones(hamming_code.shape[1])
        correction = bp.decode(syndrome, llr, 30)
        assert np.all(correction == x)
