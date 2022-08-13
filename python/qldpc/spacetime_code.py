from typing import Callable, Iterable, Tuple, Dict, List, Deque
from dataclasses import dataclass
import scipy.sparse as sparse
import numpy as np
from itertools import repeat
from functools import partial

@dataclass
class SpacetimeCodeSingleShot:
    check_matrix : sparse.spmatrix
    final_correction : Callable[[np.array], np.array]

def single_shot(check_matrix : sparse.spmatrix) -> (sparse.spmatrix, Callable[[np.array], np.array]):
    '''Construct a check matrix that is extended for measurement errors.
    We append columns to check matrix to extend the support of each check to a new bit.
    Returns projection of correction to data bits
    '''

    # Extend check matrix H to (H|I)
    # The extra bits hanging off the end correspond to measurement failures
    extended_check_matrix = sparse.hstack([check_matrix, sparse.identity(check_matrix.shape[0], dtype=check_matrix.dtype)])
    # Project to the original data bits
    projection = lambda x, n=check_matrix.shape[1]: x[:n]
    return SpacetimeCodeSingleShot(extended_check_matrix, projection)

@dataclass
class SpacetimeCode:
    check_matrix : sparse.spmatrix
    syndrome_from_history : Callable[[Callable[[int], np.array], np.array], np.array]
    final_correction : Callable[[Callable[[int], np.array]], np.array]

def spacetime_code(check_matrix : sparse.spmatrix, num_rounds : int) -> SpacetimeCode:
    '''Construct a check matrix that is the corresponding space-time code that localizes errors to single points in a syndrome history.
    (tracking syndrome differences)'''

    # Stack copies of the checks
    check_matrix_coo = check_matrix.tocoo()
    spacetime_check_matrix = sparse.block_diag(repeat(check_matrix_coo, num_rounds+1))

    data_block_size = spacetime_check_matrix.shape[1]
    r = check_matrix.shape[0]

    # Add measurement failure bits    
    measurement_block_i = np.zeros(num_rounds*r*2)
    measurement_block_j = np.zeros(num_rounds*r*2)

    for i, pair in enumerate((i*r + j, (i+1)*r + j) for i in range(num_rounds) for j in range(r)):
        x1, x2 = pair
        measurement_block_i[i*2  ] = x1
        measurement_block_j[i*2  ] = i
        
        measurement_block_i[i*2+1] = x2
        measurement_block_j[i*2+1] = i

    measurement_block = sparse.coo_matrix((np.ones_like(measurement_block_i), (measurement_block_i, measurement_block_j)),
        shape=((num_rounds+1)*r, num_rounds*r), dtype=np.uint32)
    spacetime_check_matrix = sparse.hstack([spacetime_check_matrix, measurement_block])

    syndrome_from_history = partial(spacetime_syndrome, rounds=num_rounds, check_matrix=check_matrix)
    return SpacetimeCode(spacetime_check_matrix, syndrome_from_history, partial(correction_from_spacetime_correction, rounds=num_rounds))

def spacetime_code_sliding_window(check_matrix : sparse.spmatrix, num_window_size : int) -> sparse.spmatrix:
    pass

def spacetime_syndrome(rounds : int, check_matrix : sparse.spmatrix, syndrome_history : Callable[[int], np.array], readout : np.array) -> np.array:
    '''Convert a syndrome history + transverse readout into a syndrome for the spacetime code'''
    # number of checks
    r = check_matrix.shape[0]
    syndrome = np.zeros((rounds+1)*r)

    # Get syndrome
    for i in range(0, rounds):
        syndrome[r*i:r*(i+1)] = syndrome_history(i)

    # Last round syndrome
    syndrome[r*rounds:r*(rounds+1)] = (check_matrix @ readout)%2
    
    # Take differences
    # Each row is a timestep
    syndrome_matrix = syndrome.reshape((rounds+1, r), order='C')
    # Do this in two steps to avoid mutation problems wit
    diff = (syndrome_matrix[1:rounds+1, :] + syndrome_matrix[0:rounds, :])%2
    syndrome_matrix[1:rounds+1, :] = diff
    syndrome = syndrome_matrix.reshape((rounds+1)*r, order='C')
    
    return syndrome
    
def correction_from_spacetime_correction(rounds : int, spacetime_correction : Callable[[int], np.array]) -> np.array:
    '''Convert a correction of the spacetime code to a correction to the final round'''
    return sum(map(spacetime_correction, range(rounds+1)))%2
    
