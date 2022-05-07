from .qecc_util import GF2
import numpy as np
from numba import njit

def gf2_smith_normal_form(A: np.array) -> (np.array, np.array, np.array):
    '''Returns the Smith normal form of D=SAT. Based on the Galois decomposition routines'''
    D = A.copy()
    S = GF2.Identity(A.shape[0])
    T = GF2.Identity(A.shape[1])

    row_reduce_coeffs = GF2.Zeros(A.shape[0])

    # row reduce 
    pivot = 0
    for j in range(D.shape[1]):
        nonzero_idx = np.nonzero(D[pivot:, j])[0]
        # No pivot in this column
        if nonzero_idx.size == 0:
            continue

        i = pivot + nonzero_idx[0]

        if pivot != i:
            # D[i,j] is pivot so row swap to fix it
            D[[pivot,i], :] = D[[i,pivot], :]
            S[[pivot,i], :] = S[[i,pivot], :]

        # Add row to other rows
        row_reduce_coeffs[:] = D[:, j]
        row_reduce_coeffs[pivot] = 0

        D[:,:] -= np.multiply.outer(row_reduce_coeffs, D[pivot,:])
        S[:,:] += np.multiply.outer(row_reduce_coeffs, S[pivot,:])
        
        # exit condition
        pivot += 1
        if pivot == D.shape[0]:
            break

    # Permute columns to put 1s on diagonal
    for i in range(D.shape[0]):
        nonzero_idx = np.nonzero(D[i, :])[0]
        if nonzero_idx.size == 0:
            break
        pivot = nonzero_idx[0]

        # Swap current column with pivot column
        if pivot != i:
            D[:, [pivot, i]] = D[:, [i,pivot]]
            T[:, [pivot, i]] = T[:, [i,pivot]]
    
    return (D, S, T)
    

def gf2_row_reduce(A : np.array) -> [int]:
    '''Put a matrix in reduced row echeleon form and return the pivot columns'''
    A = A.view(GF2)
    assert(len(A.shape) == 2)
    num_rows, num_cols = A.shape

    # Columns containing the pivots
    pivot_cols = []
    # Row to put the pivot on
    r = 0
    # Put in row echeleon form
    for k in range(A.shape[1]):
        nonzeros = np.nonzero(A[r:, k])[0]
        if nonzeros.shape[0] == 0:
            continue
        
        pivot_idx = nonzeros[0]+r

        # Swap pivot and kth row
        if pivot_idx != r:
            A[(pivot_idx, r), r:] = A[(r, pivot_idx), r:]
            pivot_idx = r
        
        pivot_cols.append(k)
        # Perform row reduction op
        pivot_row = np.reshape(A[pivot_idx, k:], (1, num_cols-k))
        column_coeffs = np.reshape(A[pivot_idx+1:, k], (num_rows-pivot_idx-1, 1))

        update = column_coeffs @ pivot_row
        A[pivot_idx+1:, k:] += update
        r += 1
        if r == num_rows:
            break

    # Reduced row echeleon form
    for row, pivot_col in reversed(list(enumerate(pivot_cols))):
        column_coeffs = np.reshape(A[:row, pivot_col], (row, 1))
        pivot_row = np.reshape(A[row, pivot_col:], (1, A.shape[1] - pivot_col))

        update = column_coeffs @ pivot_row
        A[:row, pivot_col:] += update

    return pivot_cols

def get_rank(A : np.array) -> int:
    return np.linalg.matrix_rank(GF2(A))
    
def get_generator_matrix(H : np.array) -> np.array:
    '''Returns a full rank encoding map G that satisfies HG^T = 0'''
    rr_H = np.copy(H)
    pivots = gf2_row_reduce(rr_H)
    # Remove empty rows in H
    rr_H = rr_H[~np.all(rr_H==0, axis=1)]

    # Get the columns that are not pivots
    pivot_set = set(pivots)
    not_pivots = [i for i in range(rr_H.shape[1]) if i not in pivot_set]

    G = np.eye(rr_H.shape[1] - rr_H.shape[0], rr_H.shape[1], dtype=np.uint8)

    if G.shape[0] > 0 and G.shape[0] != G.shape[1]:
        standard_col_indices = not_pivots+pivots
        rr_H_standard = rr_H[:,standard_col_indices]
        G[:, G.shape[0]:] = rr_H[:, not_pivots].transpose()
        # This assignment clobbers G without the intermediate
        tempG = np.copy(G)
        G[:, standard_col_indices] = tempG
    
    return G


def test_gf2_smith_normal_form():
    for rows in [4, 8, 16, 32]:
        for cols in [4, 8, 16, 32]:
            for _ in range(200):
                A = GF2(np.where(np.random.rand(rows, cols) < 0.3, 1, 0))
                D, S, T = gf2_smith_normal_form(A)

                assert np.all(D == S@A@T)

def test_gf2_row_reduce():
    for rows in [2, 4, 8, 16, 32]:
        for cols in [2, 4, 8, 16, 32]:
            for _ in range(200):
                A = np.where(np.random.rand(rows, cols) < 0.3, 1, 0)
                pivot_cols = gf2_row_reduce(A)

                # Check reduced row echeleon
                last_pivot_col = None
                for i in range(rows):
                    nonzeros = np.nonzero(A[i, :])[0]

                    if nonzeros.shape[0] == 0:
                        last_pivot_col = cols-1
                    else:
                        next_pivot_col = nonzeros[0]
                        # The pivot should be the only 1 in the column
                        assert(len(np.nonzero(A[:, next_pivot_col])[0]) == 1)
                        if last_pivot_col is not None:
                            assert(next_pivot_col > last_pivot_col)
                        last_pivot_col = next_pivot_col

                # Check pivot columns contain only a single 1
                for i,col in enumerate(pivot_cols):
                    assert(np.nonzero(A[:, col])[0] == [i])

def test_generator_matrix():
    for rows in [2, 4, 8, 16, 32]:
        for cols in [4, 8, 16, 32]:
            for _ in range(200):
                H = np.where(np.random.rand(rows, cols) < 0.3, 1, 0)
                G = get_generator_matrix(H)
                
                # G maps to kernel H
                assert(np.all(H.view(GF2) @ G.transpose().view(GF2) == 0))
                # G is full rank
                pivots = gf2_row_reduce(G.transpose())
                assert(len(pivots) == G.shape[0])


                
