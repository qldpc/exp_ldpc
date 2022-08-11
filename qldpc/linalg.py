from .qecc_util import GF2
import numpy as np
# from numba import njit

# We use routines from galois now

# @njit
# def row_swap(A: np.array, i, j):
#     '''Swap rows i and j in A'''
#     for k in range(A.shape[1]):
#         t = A[i,k]
#         A[i,k] = A[j,k]
#         A[j,k] = t

# @njit
# def col_swap(A: np.array, i, j):
#     '''Swap columns i and j in A'''
#     for k in range(A.shape[0]):
#         t = A[k,i]
#         A[k,i] = A[k,j]
#         A[k,j] = t

# def gf2_smith_normal_form(A: np.array) -> (np.array, np.array, np.array):
#     '''Returns the Smith normal form of D=SAT. Based on the Galois decomposition routines'''
#     # Wrapper to convert to regular numpy array if it's GF2
#     snf = _gf2_smith_normal_form(np.array(A).astype(np.int8))
#     # Convert back if we started with GF2
#     if type(A) is GF2:
#         snf = tuple(GF2(x) for x in snf)
#     return snf

# @njit
# def _gf2_smith_normal_form(A: np.array) -> (np.array, np.array, np.array):
#     '''Returns the Smith normal form of D=SAT. Based on the Galois decomposition routines'''
#     D = A.copy()
#     S = np.eye(A.shape[0], dtype=np.int8)
#     T = np.eye(A.shape[1], dtype=np.int8)

#     row_reduce_coeffs = np.zeros(D.shape[0], dtype=np.int8)

#     # row reduce 
#     pivot = 0
#     for j in range(D.shape[1]):
#         nonzero_idx = np.nonzero(D[pivot:, j])[0]
#         # No pivot in this column
#         if nonzero_idx.size == 0:
#             continue

#         i = pivot + nonzero_idx[0]

#         # D[i,j] is pivot so row swap to fix it
#         if pivot != i:
#             row_swap(D, pivot, i)
#             row_swap(S, pivot, i)

#         # Add row to other rows
#         row_reduce_coeffs[:] = D[:, j]
#         row_reduce_coeffs[pivot] = 0

#         D[:,:] = (D[:,:] + np.outer(row_reduce_coeffs, D[pivot,:]))%2
#         S[:,:] = (S[:,:] + np.outer(row_reduce_coeffs, S[pivot,:]))%2
        
#         # exit condition
#         pivot += 1
#         if pivot == D.shape[0]:
#             break

#     col_reduce_coeffs = np.zeros(D.shape[1], dtype=np.int8)
#     # Permute columns to put 1s on diagonal
#     # TODO: handle junk on the right side for short / wide matrix
#     for i in range(D.shape[0]):
#         nonzero_idx = np.nonzero(D[i, :])[0]
#         if nonzero_idx.size == 0:
#             break
#         pivot = nonzero_idx[0]

#         # Swap current column with pivot column
#         if pivot != i:
#             col_swap(D, pivot, i)
#             col_swap(T, pivot, i)

#         # Reduce columns
#         col_reduce_coeffs[:] = D[i, :]
#         col_reduce_coeffs[i] = 0

#         D[:,:] = (D[:,:] + np.outer(D[:,i], col_reduce_coeffs))%2
#         T[:,:] = (T[:,:] + np.outer(T[:,i], col_reduce_coeffs))%2

#     return (D, S, T)


def gf2_get_pivots(A : np.array) -> [int]:
    largest_index = (A!=0).argmax(axis=1)
    return np.extract(A[range(A.shape[0]), largest_index]!=0, largest_index)


def get_rank(A : np.array) -> int:
    return np.linalg.matrix_rank(GF2(A))
     
# def test_gf2_smith_normal_form():
#     for rows in [4, 8, 16, 32, 64]:
#         for cols in [4, 8, 16, 32, 64]:
#             for _ in range(200):
#                 A = np.where(np.random.rand(rows, cols) < 0.3, 1, 0)
#                 D, S, T = gf2_smith_normal_form(A)

#                 assert np.all(D == GF2(S)@GF2(A)@GF2(T))

