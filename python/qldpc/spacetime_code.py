from typing import Callable, Iterable, Tuple, Dict, List, Deque
from dataclasses import dataclass
import scipy.sparse as sparse
import numpy as np
import stim
from itertools import repeat, chain
from functools import partial
from numpy.typing import ArrayLike

@dataclass(frozen=True)
class SpacetimeCodeSingleShot:
    spacetime_check_matrix : sparse.spmatrix
    _datablock_size : int

    def __init__(self, check_matrix : sparse.spmatrix):
        '''Construct a check matrix that is extended for measurement errors.
        We append columns to check matrix to extend the support of each check to a new bit.
        '''

        # Extend check matrix H to (H|I)
        # The extra bits hanging off the end correspond to measurement failures
        extended_check_matrix = sparse.hstack([check_matrix, sparse.identity(check_matrix.shape[0], dtype=check_matrix.dtype)])
        # Project to the original data bits
        object.__setattr__(self, '_datablock_size', check_matrix.shape[1])
        object.__setattr__(self, 'spacetime_check_matrix', extended_check_matrix)
    
    def final_correction(self, x : ArrayLike) -> ArrayLike:
        '''Get the final round correction'''
        return self.data_bits(x)
    
    def data_bits(self, x : ArrayLike) -> ArrayLike:
        '''View to the data bits of the spacetime vector'''
        return x[:self._datablock_size]
    
    def measurement_bits(self, x : ArrayLike) -> ArrayLike:
        '''View to the measurement bits of the spacetime vector'''
        return x[self._datablock_size:]

@dataclass(frozen=True)
class SpacetimeCode:
    spacetime_check_matrix : sparse.spmatrix
    inactivation_sets : sparse.spmatrix
    _check_matrix : sparse.spmatrix
    _num_rounds : int
    _datablock_size : int

    def __init__(self, check_matrix : sparse.spmatrix, num_rounds : int, dual_basis_checks : sparse.spmatrix = None):
        '''Construct a check matrix that is the corresponding space-time code that localizes errors to single points in a syndrome history.
        (tracking syndrome differences)'''

        # Stack copies of the checks
        check_matrix_coo = check_matrix.tocoo()
        spacetime_check_matrix = sparse.block_diag(repeat(check_matrix_coo, num_rounds+1))

        r = check_matrix.shape[0]

        # Add measurement failure bits    
        measurement_block_i = np.zeros(num_rounds*r*2, dtype=np.uint32)
        measurement_block_j = np.zeros(num_rounds*r*2, dtype=np.uint32)

        for i, pair in enumerate((i*r + j, (i+1)*r + j) for i in range(num_rounds) for j in range(r)):
            x1, x2 = pair
            measurement_block_i[i*2  ] = x1
            measurement_block_j[i*2  ] = i
            
            measurement_block_i[i*2+1] = x2
            measurement_block_j[i*2+1] = i

        measurement_block = sparse.coo_matrix((np.ones_like(measurement_block_i), (measurement_block_i, measurement_block_j)),
            shape=((num_rounds+1)*r, num_rounds*r), dtype=np.uint32)
        spacetime_check_matrix = sparse.hstack([spacetime_check_matrix, measurement_block]).tocsr()

        object.__setattr__(self, '_check_matrix', check_matrix)
        object.__setattr__(self, 'spacetime_check_matrix', spacetime_check_matrix)
        object.__setattr__(self, '_num_rounds', num_rounds)
        object.__setattr__(self, '_datablock_size', measurement_block.shape[1])
        object.__setattr__(self, 'inactivation_sets', None)

        if dual_basis_checks is not None:
            # Construct inactivation_sets by stacking Z checks in time
            inactivation_sets = sparse.hstack([
                sparse.block_diag(repeat(dual_basis_checks.tocoo(), num_rounds+1)),
                sparse.coo_matrix(([], ([], [])), shape=measurement_block.shape)]).tocsr()
            assert inactivation_sets.shape[1] == self.spacetime_check_matrix.shape[1]
            object.__setattr__(self, 'inactivation_sets', inactivation_sets)

    def syndrome_from_history(self, history : Callable[[int], ArrayLike], readout : ArrayLike) -> ArrayLike:
        '''Convert a history of measurements to a spacetime syndrome'''
        return _spacetime_syndrome(self._num_rounds, self._check_matrix, history, readout)

    def final_correction(self, spacetime_correction : ArrayLike) -> ArrayLike:
        '''Convert a correction of the spacetime code to a correction to the final round'''
        num_qubits = self._check_matrix.shape[1]
        return sum(spacetime_correction[i*num_qubits : (i+1)*num_qubits] for i in range(self._num_rounds+1))%2

    def data_bits(self, x : ArrayLike) -> ArrayLike:
        '''View to the data bits of the spacetime vector'''
        return x[:self._datablock_size]
    
    def measurement_bits(self, x : ArrayLike) -> ArrayLike:
        '''View to the measurement bits of the spacetime vector'''
        return x[self._datablock_size:]


# def spacetime_code_sliding_window(check_matrix : sparse.spmatrix, num_window_size : int) -> sparse.spmatrix:
#     pass

def _spacetime_syndrome(rounds : int, check_matrix : sparse.spmatrix, syndrome_history : Callable[[int], np.array], readout : np.array) -> np.array:
    '''Convert a syndrome history + transverse readout into a syndrome for the spacetime code'''
    # number of checks
    r = check_matrix.shape[0]
    syndrome = np.zeros((rounds+1)*r, dtype=np.uint32)

    # Get syndrome
    for i in range(0, rounds):
        syndrome[r*i:r*(i+1)] = syndrome_history(i)

    # Last round syndrome
    syndrome[r*rounds:r*(rounds+1)] = (check_matrix @ readout)%2
    
    # Take differences
    # Each row is a timestep
    syndrome_matrix = syndrome.reshape((rounds+1, r), order='C')
    # Do this in two steps to avoid mutation problems
    diff = (syndrome_matrix[1:rounds+1, :] + syndrome_matrix[0:rounds, :])%2
    syndrome_matrix[1:rounds+1, :] = diff
    syndrome = syndrome_matrix.reshape((rounds+1)*r, order='C')
    
    return syndrome


@dataclass(frozen=True)
class DetectorSpacetimeCode:
    '''Variant of spacetime code that starts from a Stim detector model and produces an effective "Fault check matrix", fault priors, and fault mapping matrix'''

    fault_check_matrix : sparse.spmatrix
    fault_map : np.array
    fault_priors : np.array

    def __init__(self, detector_model : stim.DetectorErrorModel):
        detector_model = detector_model.flattened()
        
        detector_offset = 0
        fault_check_matrix_cols = []
        fault_map_cols = []
        fault_priors = []

        max_logical_idx = -1
        max_detector_idx = -1
        for instruction in detector_model:
            # Requires a flattened model for now
            assert any(instruction.type == t for t in ['error', 'detector', 'logical_observable'])

            # These may be empty
            try:
                max_detector_idx = max(max_detector_idx, max(x.val+detector_offset for x in instruction.targets_copy() if x.is_relative_detector_id()))
            except ValueError:
                pass

            try:
                max_logical_idx = max(max_logical_idx, max(x.val for x in instruction.targets_copy() if x.is_logical_observable_id()))
            except ValueError:
                pass

            if instruction.type == 'error':
                error = instruction
                detectors = [x.val + detector_offset for x in error.targets_copy() if x.is_relative_detector_id()]
                logicals = [x.val for x in error.targets_copy() if x.is_logical_observable_id()]
                assert len(error.args_copy()) == 1

                fault_check_matrix_cols.append(detectors)
                fault_map_cols.append(logicals)
                fault_priors.append(error.args_copy()[0])

        num_detectors = 1+max_detector_idx
        num_logicals = 1+max_logical_idx

        def lil_to_coo(a):
            for j, col in enumerate(a):
                for i, entry in enumerate(col):
                    yield [i,j,1]
                    
        def mk_sparse(a, shape):
            IJV_entries = list(lil_to_coo(a))
            IJV = np.array(IJV_entries, dtype=np.uint32)
            return sparse.coo_matrix((IJV[:,2], (IJV[:,0], IJV[:,1])), shape=shape, dtype=np.uint32).tocsr()

        fault_check_matrix = mk_sparse(fault_check_matrix_cols, shape=(num_detectors, len(fault_check_matrix_cols)))
        fault_map = mk_sparse(fault_map_cols, shape=(num_logicals, len(fault_map_cols)))
        fault_priors = np.array(fault_priors)
        object.__setattr__(self, 'fault_check_matrix', fault_check_matrix)
        object.__setattr__(self, 'fault_map', fault_map)
        object.__setattr__(self, 'fault_priors', fault_priors)
    
    
