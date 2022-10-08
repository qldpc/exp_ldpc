import scipy.sparse as sparse
import numpy as np
from numba import int32, float64
from typing import Dict
from array import array

def _log1pexp(x):
    '''Compute Log[1+Exp[x]] without overflowing in the exp.
    The discontinuity at 12e is about 6.8e-15'''
    assert not np.isnan(x)
    
    transition = 12*np.e
    return x if x > transition else np.log1p(np.exp(x))

class BeliefPropagation:
    _check_to_bit : Dict
    _bit_to_check : Dict
    _check_matrix : sparse.csr_matrix
    
    def __init__(self, check_matrix : sparse.spmatrix):
        self._check_to_bit = {i:array('L') for i in range(check_matrix.shape[0])}
        self._bit_to_check = {j:array('L') for j in range(check_matrix.shape[1])}

        for i,j in zip(*check_matrix.nonzero()):
            self._check_to_bit[i].append(j)
            self._bit_to_check[j].append(i)
        
        self._check_matrix = check_matrix.tocsr()
    

    @staticmethod
    def llr_sum(a_llr, b_llr):
        '''Numerically stable way to compute the llr of (a + b). Equation (6) of Chen et al. IEEE Trans. Comm. 53 (8) 1288-1299 (2005)'''
        ab_sum = np.abs(a_llr + b_llr)
        ab_diff = np.abs(a_llr - b_llr)
        
        return (
            np.sign(a_llr) * np.sign(b_llr) * np.minimum(np.abs(a_llr), np.abs(b_llr))
            + _log1pexp(ab_sum)
            - _log1pexp(ab_diff)
        )

    def check_converged(self, llr, syndrome):
        return np.all((self._check_matrix @ np.where(llr > 0, 1, 0).astype(np.uint32))%2 == syndrome)

    def decode(self, syndrome, llr_prior, iterations, harden=None):
        if harden is None:
            harden = True
        
        syndrome_R = np.where(syndrome == 0, 1, -1)
        # We can allocate this as a skinny matrix with row/col indices given by local coordinates instead of global ones
        # Take max local system size to make everything convenient
        messages_v_to_c = np.zeros((len(syndrome), len(llr_prior)))
        messages_c_to_v = np.zeros((len(syndrome), len(llr_prior)))
        
        llr = llr_prior
        for _ in range(iterations):
            # v -> c
            for bit, check_list in self._bit_to_check.items():
                for check in check_list:
                    messages_v_to_c[check,bit] = llr[bit] - messages_c_to_v[check, bit]

            # c -> v
            for check, bit_list in self._check_to_bit.items():
                # We need to handle the case where check_list has 1 element
                # Jacobian update from Chen et al. (2005)
                # Forward computation
                f_llr = np.zeros(len(bit_list))
                f_llr[0] = messages_v_to_c[check,bit_list[0]]
                for i in range(1, len(bit_list)):
                    f_llr[i] = self.llr_sum(f_llr[i-1], messages_v_to_c[check, bit_list[i]])

                # Backwards computation
                b_llr = np.zeros(len(bit_list))
                b_llr[-1] = messages_v_to_c[check, bit_list[-1]]
                for i in range(len(bit_list)-2, -1, -1):
                    b_llr[i] = self.llr_sum(b_llr[i+1], messages_v_to_c[check, bit_list[i]])

                # Compute messages
                messages_c_to_v[check, bit_list[0]] = syndrome_R[check]*b_llr[1]
                messages_c_to_v[check, bit_list[-1]] = syndrome_R[check]*f_llr[-2]
                for i in range(2, len(bit_list)-1):
                    messages_c_to_v[check, bit_list[i]] = syndrome_R[check]*self.llr_sum(f_llr[i-1], b_llr[i+1])
                    

            # LLRs
            llr = np.copy(llr_prior)
            for bit, check_list in self._bit_to_check.items():
                llr[bit] += sum(messages_c_to_v[c, bit] for c in check_list)

            if self.check_converged(llr, syndrome):
                break
                    
        return np.where(llr > 0, 1, 0).astype(np.uint32) if harden else llr
