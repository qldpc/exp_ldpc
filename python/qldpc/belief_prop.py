import scipy.sparse as sparse
import numpy as np
from numba import int32, float64
from typing import Dict
from array import array

class BeliefPropagation:
    _check_to_bit : Dict
    _bit_to_check : Dict
    
    def __init__(self, check_matrix : sparse.spmatrix):
        self._check_to_bit = {i:array('L') for i in range(check_matrix.shape[0])}
        self._bit_to_check = {j:array('L') for j in range(check_matrix.shape[1])}

        for i,j in zip(*check_matrix.nonzero()):
            self._check_to_bit[i].append(j)
            self._bit_to_check[j].append(i)

            
    @staticmethod
    def llr_sum(a_llr, b_llr):
        '''Numerically stable way to compute the llr of (a + b). Equation (6) of Chen et al. IEEE Trans. Comm. 53 (8) 1288-1299 (2005)'''
        ab_sum = np.abs(a_llr + b_llr)
        ab_diff = np.abs(a_llr - b_llr)
        
        return (
            np.sign(a_llr) * np.sign(b_llr) * np.minimum(np.abs(a_llr), np.abs(b_llr))
            + np.log1p(np.exp(ab_sum))
            - np.log1p(np.exp(ab_diff))
        )

    def decode(self, syndrome, llr_prior, iterations, harden=None):
        if harden is None:
            harden = True
        
        syndrome_R = np.where(syndrome == 0, 1, -1)
        messages_v_to_c = np.zeros((len(syndrome), len(llr_prior)))
        messages_c_to_v = np.zeros((len(syndrome), len(llr_prior)))
        
        llr = llr_prior
        for _ in range(iterations):
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
                messages_c_to_v[check, bit_list[0]] = b_llr[1]
                messages_c_to_v[check, bit_list[-1]] = f_llr[-2]
                for i in range(2, len(bit_list)-1):
                    messages_c_to_v[check, bit_list[i]] = self.llr_sum(f_llr[i-1], b_llr[i+1])
                    
            print(f'{messages_c_to_v=}')

            # LLRs
            llr = llr_prior
            for bit, check_list in self._bit_to_check.items():
                llr[bit] = np.sum(messages_c_to_v[c, bit] for c in check_list)

            # v -> c
            for bit, check_list in self._bit_to_check.items():
                for check in check_list:
                    messages_v_to_c[check,bit] = llr[bit] - messages_c_to_v[check, bit]

            print(llr)
                    
        return np.where(llr < 0, 1, 0).astype(np.uint32) if harden else llr
