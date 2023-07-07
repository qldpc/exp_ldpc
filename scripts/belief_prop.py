import scipy.sparse as sparse
import numpy as np
import numba
from numba import njit
from typing import Dict
from array import array

@njit(numba.float64(numba.float64), inline='always')
def _log1pexp(x):
    '''Compute Log[1+Exp[x]] without overflowing in the exp.
    The discontinuity at 32 is about 1e-15'''
    
    transition = 32
    return x if x > transition else np.log1p(np.exp(x))

@njit(numba.float64(numba.float64, numba.float64), inline='always')
def _llr_sum(a_llr, b_llr):
    '''Numerically stable way to compute the llr of (a + b). Equation (6) of Chen et al. IEEE Trans. Comm. 53 (8) 1288-1299 (2005)'''
    ab_sum = np.abs(a_llr + b_llr)
    ab_diff = np.abs(a_llr - b_llr)
    
    return (
        np.sign(a_llr) * np.sign(b_llr) * np.minimum(np.abs(a_llr), np.abs(b_llr))
        + _log1pexp(-ab_sum)
        - _log1pexp(-ab_diff)
    )

@njit(numba.int32(numba.int32[:], numba.int32, numba.int32), inline='always')
def scan_idx(a, n : int, j : int) -> int:
    '''Return i s.t. a[i] = j. Behavior undefined if no such entry exists. n is the length of a'''
    for i in range(n):
        if a[i] == j:
            return i
    return -1

@njit(numba.int32(numba.int32[:], numba.int32), inline='always')
def scan_deg(a, n : int) -> int:
    '''Return a past-the-end index for range of valid entries'''
    for i in range(n):
        if a[i] < 0:
            return i
    else:
        return n

class BeliefPropagation:
    _check_to_bit : np.array
    _bit_to_check : np.array
    _check_matrix : sparse.csr_matrix

    @staticmethod
    def _dok_to_adjacency(a : Dict):
        '''Convert a dictionary of keys adjacency map to an array where along each row is the adjacent items and -1 is used to indicate None'''
        max_degree = max(len(x) for x in a.values())
        adjacency = -np.ones((len(a), max_degree), order='C', dtype=np.int32)
        for i, adj in a.items():
            for j in range(len(adj)):
                adjacency[i,j] = adj[j]
        return adjacency

    def __init__(self, check_matrix : sparse.spmatrix):
        check_to_bit = {i:array('L') for i in range(check_matrix.shape[0])}
        bit_to_check = {j:array('L') for j in range(check_matrix.shape[1])}

        for i,j in zip(*check_matrix.nonzero()):
            check_to_bit[i].append(j)
            bit_to_check[j].append(i)

        self._check_to_bit = self._dok_to_adjacency(check_to_bit)
        self._bit_to_check = self._dok_to_adjacency(bit_to_check)
        
        self._check_matrix = check_matrix.tocsr()

    def harden_llr(self, llr):
        return np.where(llr > 0, 0, 1).astype(np.uint32)
        
    def check_converged(self, llr, syndrome):
        return np.all((self._check_matrix @ self.harden_llr(llr))%2 == syndrome)

    def decode(self, syndrome, llr_prior, iterations, harden=None, clamp_llr=None, break_converged=None,):
        # TODO: Specialize on the check/bit degree so we can unroll these loops
        # We can allocate this as a skinny matrix with row/col indices given by local coordinates instead of global ones
        # Take max local system size to make everything convenient
        # TODO: we can reorganize these loops to avoid the scan_idx by accumulating messages at the target
        # For cache locality purposes, we could split the variables into sets and we update the sets a few times before moving to the next one
        # Ex. we hammer a single time slice at time

        # Max magnitude we permit the check to bit messages to have
        # This is necessary because the LLRs will run to +/- inf after BP converges
        # Once that happens, eventually the update rule will start to run into round-off error and the updates will oscillate wildly
        if clamp_llr is None:
            clamp_llr = np.inf

        if harden is None:
            harden = True

        if break_converged is None:
            break_converged = False

        syndrome_R = np.where(syndrome == 0, 1, -1)

        max_bit_degree = self._bit_to_check.shape[1]
        max_check_degree = self._check_to_bit.shape[1]

        # ==== initialize ====
        f_llr = np.zeros(max_check_degree, dtype=np.float64)
        b_llr = np.zeros(max_check_degree, dtype=np.float64)
        gathered_m_v_to_c = np.zeros(max_check_degree, dtype=np.float64)
        
        # The messages here are stored at the source
        # We would eventually like to store them at the target
        messages_v_to_c = np.zeros_like(self._bit_to_check, dtype=np.float64)
        messages_c_to_v = np.zeros_like(self._check_to_bit, dtype=np.float64)

        llr = np.copy(llr_prior)
        
        # ==== BP iterations ====
        for _ in range(iterations):
            bpd_iteration_jit(syndrome_R=syndrome_R, clamp_llr=clamp_llr, llr_prior=llr_prior,
                              bit_to_check=self._bit_to_check, check_to_bit=self._check_to_bit, # Normally this would be a class member
                              f_llr=f_llr, b_llr=b_llr, gathered_m_v_to_c=gathered_m_v_to_c,
                              messages_v_to_c=messages_v_to_c, messages_c_to_v=messages_c_to_v,
                              llr=llr)
            if break_converged and self.check_converged(llr, syndrome):
                break
        
        return self.harden_llr(llr) if harden else llr

@njit
def bpd_iteration_jit(syndrome_R, clamp_llr, llr_prior,
                      bit_to_check, check_to_bit,
                      f_llr, b_llr, gathered_m_v_to_c, messages_v_to_c, messages_c_to_v,
                      llr):
    
    num_bits = numba.int32(bit_to_check.shape[0])
    num_checks = numba.int32(check_to_bit.shape[0])
    max_bit_degree = numba.int32(bit_to_check.shape[1])
    max_check_degree = numba.int32(check_to_bit.shape[1])
    
    # ===== v -> c =====
    for bit in range(num_bits):
        # Step along the row in the bit to check adjacency structure
        bit_degree = scan_deg(bit_to_check[bit, :], max_bit_degree)

        # Sum all the message going into bit
        message_sum = llr[bit]
        for check_j in range(bit_degree):
            check = bit_to_check[bit, check_j]
            # Find the index of the bit in the check_to_bit adjacency structure
            bit_j = scan_idx(check_to_bit[check, :], max_check_degree, bit)
            
            message_sum += messages_c_to_v[check, bit_j]

        # Compute outgoing messages
        for check_j in range(bit_degree):
            check = bit_to_check[bit, check_j]
            # Find the index of the bit in the check_to_bit adjacency structure
            bit_j = scan_idx(check_to_bit[check, :], max_check_degree, bit)

            # Compute message
            messages_v_to_c[bit, check_j] = message_sum - messages_c_to_v[check, bit_j]

    # ===== c -> v =====
    for check in range(num_checks):
        # We need to handle the case where check_list has 1 element
        gathered_m_v_to_c[:] = np.nan
        check_degree = scan_deg(check_to_bit[check,:], max_check_degree)

        # ====== Gather incoming messages ======
        for bit_j in range(max_check_degree):
            # Look up incoming message
            bit = check_to_bit[check,bit_j]
            check_j = scan_idx(bit_to_check[bit, :], max_bit_degree, check)
            gathered_m_v_to_c[bit_j] = messages_v_to_c[bit, check_j]

        # ====== Standard product sum update ======
        # for bit_j in range(0, check_degree):
        #     gathered_m_v_to_c[bit_j] = np.tanh(gathered_m_v_to_c[bit_j]/2)

        # for bit_j in range(0, check_degree):
        #     prod = 1
        #     for bit_i in range(0,check_degree):
        #         if bit_i != bit_j:
        #             prod *= gathered_m_v_to_c[bit_i]
        #     messages_c_to_v[check, bit_j] = syndrome_R[check]*2*np.arctanh(prod)
            
        # ====== Jacobian update from Chen et al. (2005) ======
        # Forward computation
        f_llr[:] = np.nan
        f_llr[0] = gathered_m_v_to_c[0]
        for bit_j in range(1, check_degree):
            # (Recursive) forward computation rule
            f_llr[bit_j] = _llr_sum(f_llr[bit_j-1], gathered_m_v_to_c[bit_j])

        # Backwards computation
        b_llr[:] = np.nan
        b_llr[check_degree-1] = gathered_m_v_to_c[check_degree-1]
        for bit_j_end_offset in range(1, check_degree):
            bit_j = check_degree - bit_j_end_offset - 1
            b_llr[bit_j] = _llr_sum(b_llr[bit_j+1], gathered_m_v_to_c[bit_j])

        # Compute messages
        messages_c_to_v[check, 0] = syndrome_R[check]*b_llr[1]
        messages_c_to_v[check, check_degree-1] = syndrome_R[check]*f_llr[check_degree-2]
        for bit_j in range(1, check_degree-1):
            messages_c_to_v[check, bit_j] = syndrome_R[check]*_llr_sum(f_llr[bit_j-1], b_llr[bit_j+1])

        # clip message amplitude
        messages_c_to_v[check, :] = np.clip(messages_c_to_v[check, :], -clamp_llr, clamp_llr)

    # ===== LLRs =====

    # Initialize with prior
    for bit in range(num_bits):
        llr[bit] = llr_prior[bit]

    # Update priors with passed messages
    for check in range(num_checks):
        for bit_j in range(max_check_degree):
            bit = check_to_bit[check, bit_j]
            if bit >= 0:
                llr[bit] += messages_c_to_v[check, bit_j]
