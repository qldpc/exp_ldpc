from qldpc import BeliefPropagation
import numpy as np
import scipy.sparse as sparse

# A small LDPC code from MacKay's website
# 96.3.963 (N=96,K=48,M=48,R=0.5)
_checks =  [[10, 30, 40], [5, 32, 45], [16, 18, 39], [12, 22, 38], [15, 19, 47], [2, 17, 34], [9, 24, 42], 
    [1, 29, 33], [4, 27, 36], [3, 26, 35], [11, 31, 43], [7, 21, 44], [8, 20, 48], [14, 23, 46], 
    [6, 28, 37], [13, 25, 41], [14, 32, 43], [5, 23, 37], [2, 31, 36], [1, 28, 34], [7, 25, 47], 
    [10, 21, 33], [15, 30, 35], [16, 26, 48], [3, 22, 46], [12, 20, 41], [8, 18, 38], [4, 19, 45], 
    [6, 24, 40], [9, 27, 39], [13, 17, 42], [11, 29, 44], [8, 24, 34], [6, 25, 36], [9, 19, 43], 
    [1, 20, 46], [14, 27, 42], [7, 22, 39], [13, 18, 35], [4, 26, 40], [16, 29, 38], [15, 21, 48], 
    [11, 23, 45], [3, 17, 47], [5, 28, 44], [12, 32, 33], [2, 30, 41], [10, 31, 37], [10, 18, 36], 
    [4, 23, 44], [9, 29, 40], [2, 27, 38], [8, 30, 42], [12, 28, 43], [11, 20, 37], [1, 19, 35], 
    [15, 31, 39], [16, 32, 41], [5, 26, 33], [3, 25, 45], [13, 21, 34], [14, 24, 48], [7, 17, 46], 
    [6, 22, 47], [7, 27, 40], [11, 18, 33], [2, 32, 35], [10, 28, 47], [5, 24, 41], [12, 25, 37], 
    [3, 19, 39], [14, 31, 44], [16, 30, 34], [13, 20, 38], [9, 22, 36], [6, 17, 45], [4, 21, 42], 
    [15, 29, 46], [8, 26, 43], [1, 23, 48], [1, 25, 42], [15, 22, 40], [8, 21, 41], [9, 18, 47], 
    [6, 27, 43], [11, 30, 46], [7, 31, 35], [5, 20, 36], [14, 17, 38], [16, 28, 45], [4, 32, 37], 
    [13, 23, 33], [12, 26, 44], [3, 29, 48], [2, 24, 39], [10, 19, 34]]

def test_belief_prop_small():
    check_matrix = sparse.csr_matrix(np.array([[1,1,0],[0,1,1]], dtype=np.uint32))

    for j in range(check_matrix.shape[1]):
        x = np.zeros(check_matrix.shape[1], dtype=np.uint32)
        x[j] = 1

        syndrome = (check_matrix @ x)%2
        llr = -6*np.ones(check_matrix.shape[1])

        bp = BeliefPropagation(check_matrix)
        correction = bp.decode(syndrome, llr, 10)

        assert np.all(x == correction)

def test_belief_prop():
    check_matrix = sparse.dok_matrix((48, 96), dtype=np.uint32)
    for j, col in enumerate(_checks):
        for i in col:
            check_matrix[i-1,j] = 1
    check_matrix = check_matrix.tocsr()

    bp = BeliefPropagation(check_matrix)

    corrections = []
    for i in set(range(check_matrix.shape[1])):
        x = np.zeros(check_matrix.shape[1], dtype=np.uint32)
        x[i] = 1
        syndrome = (check_matrix @ x)%2
        llr = -6*np.ones(check_matrix.shape[1])
        correction = bp.decode(syndrome, llr, 30)
        corrections.append((i, correction, x))

    print([i for i,c,x in corrections if np.any(c != x)])

    for i, correction, x in corrections:
        assert np.all(correction == x)
