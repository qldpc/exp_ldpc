import numpy as np
from ldpc import bp_decoder, bposd_decoder
import stim
import qldpc
from qldpc import SpacetimeCode, SpacetimeCodeSingleShot, DetectorSpacetimeCode
from galois import GF2
from functools import partial
from itertools import chain
from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Optional, Set


@dataclass(frozen=True)
class BPOSDCorrectSingleShot():
    _bpd_final_round : None
    _bpd_single_shot : None
    _spacetime_code : None
    _checks : None
    _rounds : int

    def __init__(self, code : qldpc.QuantumCode, rounds : int, bp_osd_options : Dict, priors : Tuple[float, float]):
        data_prior, measurement_prior = priors

        object.__setattr__(self, '_bpd_final_round', bposd_decoder(
            code.checks.z,
            error_rate = data_prior,
            **bp_osd_options
        ))

        object.__setattr__(self, '_rounds', rounds)
        object.__setattr__(self, '_checks', code.checks.z)
        object.__setattr__(self, '_spacetime_code', SpacetimeCodeSingleShot(self._checks))

        channel_prior = np.zeros(self._spacetime_code.spacetime_check_matrix.shape[1])
        self._spacetime_code.data_bits(channel_prior)[:] = data_prior
        self._spacetime_code.measurement_bits(channel_prior)[:] = measurement_prior
    
        object.__setattr__(self, '_bpd_single_shot', bposd_decoder(
            self._spacetime_code.spacetime_check_matrix,
            channel_probs = channel_prior,
            **bp_osd_options))


    def readout_correction(self, history : Callable[[int], np.array], data_readout : np.array) -> np.array:
        accumulated_correction = np.zeros_like(data_readout, dtype=np.int32)
        for t in range(self._rounds):
            # Compute new syndrome after applying the current correction
            correction_syndrome = (self._checks @ accumulated_correction)%2
            syndrome = (correction_syndrome + history(t))%2

            # Compute correction and add it to the current correction
            correction = self._spacetime_code.final_correction(self._bpd_single_shot.decode(syndrome))
            accumulated_correction = (accumulated_correction + correction)%2

        # Correct the transverse readout
        data_readout = (accumulated_correction + data_readout)%2

        # Final round correction on read out data
        syndrome = (self._checks @ data_readout)%2
        final_correction = self._bpd_final_round.decode(syndrome)
        return (final_correction + accumulated_correction)%2

@dataclass
class BPOSDCorrect():
    _bpd : None
    _spacetime_code : SpacetimeCode
    _checks : None

    def __init__(self, code : qldpc.QuantumCodeChecks, rounds : int, bp_osd_options : Dict, priors : Tuple[float, float]):
        data_prior, measurement_prior = priors

        object.__setattr__(self, '_checks', code.checks.z)
        object.__setattr__(self, '_spacetime_code', SpacetimeCode(self._checks, rounds))

        channel_prior = np.zeros(self._spacetime_code.spacetime_check_matrix.shape[1])
        self._spacetime_code.data_bits(channel_prior)[:] = data_prior
        self._spacetime_code.measurement_bits(channel_prior)[:] = measurement_prior
        object.__setattr__(self, '_bpd', bposd_decoder(self._spacetime_code.spacetime_check_matrix, channel_prior=channel_prior, **bp_osd_options))


    def readout_correction(self, history : Callable[[int], np.array], readout : np.array) -> np.array:
        syndrome = self._spacetime_code.syndrome_from_history(history, readout)
        correction = self._bpd.decode(syndrome)
        return self._spacetime_code.final_correction(correction)

@dataclass
class SIBPDCorrection():
    _spacetime_code : SpacetimeCode
    _bpd : None
    _priors : np.array
    _si_max_iter : int

    def __init__(self, code : qldpc.QuantumCodeChecks, rounds : int, bp_options : Dict, priors : Tuple[float, float]):
        data_prior, measurement_prior = priors

        object.__setattr__(self, '_si_max_iter', bp_options['si_cutoff'])
        object.__setattr__(self, '_spacetime_code', SpacetimeCode(code.checks.z, rounds, dual_basis_checks = code.checks.x))

        

        channel_prior = np.zeros(self._spacetime_code.spacetime_check_matrix.shape[1])
        self._spacetime_code.data_bits(channel_prior)[:] = data_prior
        self._spacetime_code.measurement_bits(channel_prior)[:] = measurement_prior
        object.__setattr__(self, '_priors', channel_prior)

        
        object.__setattr__(self, '_bpd', bp_decoder(
            self._spacetime_code.spacetime_check_matrix,
            channel_probs = self._priors,
            bp_method = 'psl', max_iter = bp_options['max_iter']),
        )

    def _compute_reliability(self, posterior_llr : np.array) -> np.array:
        '''Compute reliability of stabilizer generators'''
        # This routine might need to be JIT'd
        reliability = np.zeros(self._spacetime_code.inactivation_sets.shape[0])
        row_nonzero, col_nonzero = self._spacetime_code.inactivation_sets.nonzero()
        for i in range(row_nonzero.shape[0]):
            reliability[row_nonzero[i]] += np.abs(posterior_llr[col_nonzero[i]])
        return reliability        

    @staticmethod
    def _set_to_indices(a : Set):
        '''Convert a set to a sorted numpy list'''
        return np.sort(np.fromiter(a, int, len(a)))
    
    def _check_feasible(self, syndrome, deactivated_set) -> Optional[np.array]:
        '''Returns a correction supported on deactivated_set explaining syndrome if a solution exists'''

        if len(deactivated_set) == 0:
            return (None if np.count_nonzero(syndrome) > 0
                    else np.zeros(self._spacetime_code.inactivation_sets.shape[1], dtype=np.uint32))
        
        # We need to get a submatrix of the check matrix with columns deactivated_set and rows for which there is a nonzero entry in deactivate set
        inactivation_set_csc = self._spacetime_code.inactivation_sets.tocsc()
        submatrix_rows = self._set_to_indices(set(chain(inactivation_set_csc.getcol(j).nonzero() for j in deactivated_set)))
        deactivated_set_idx = self._set_to_indices(deactivated_set)
        
        # Solve syndrome equation for the submatrix induced by the deactivated set
        check_submatrix = inactivation_set_csc[submatrix_rows, deactivated_set_idx].toarray()

        syndrome_submatrix = syndrome[submatrix_rows]
        augmented = GF2(np.hstack([check_submatrix, syndrome_submatrix[:,np.newaxis]]))
        augmented.row_reduce(ncols = min(augmented.shape[1]-1, augmented.shape[0]))
        solution_vec = augmented[:, augmented.shape[1]-1]

        if check_submatrix @ solution_vec == syndrome_submatrix:
            correction = np.zeros(self._spacetime_code.inactivation_sets.shape[1], dtype=np.uint32)
            correction[submatrix_rows] = np.array(solution_vec)
            return correction
        else:
            return None

    def readout_correction(self, history : Callable[[int], np.array], readout : np.array) -> np.array:
        syndrome = self._spacetime_code.syndrome_from_history(history, readout)

        deactivated_set = set()

        stabilizer_ranking = None
        for si_iter in range(self._si_max_iter):
            # Set prior with deactivated set
            bp_prior = self._priors
            bp_prior[self._set_to_indices(deactivated_set)] = 0.5
            self._bpd.update_channel_probs(bp_prior)
            
            # Run BP
            correction = self._bpd.decode(syndrome)

            # Check (exit condition) that the deactivated set supports a correction 
            if self._bpd.converge == 1:
                residual_syndrome = (syndrome + self._spacetime_code.spacetime_check_matrix @ correction)%2
                total_correction = self._check_feasible(residual_syndrome, deactivated_set)
                if total_correction is not None:
                    correction = total_correction
                    break

            # Compute reliability if it has not yet been computed
            if stabilizer_ranking is None:
                reliability = self._compute_reliability(self._bpd.log_prob_ratios)
                stabilizer_ranking = np.argsort(reliability)
            
            # Deactivate stabilizer at ranking si_iter
            deactivated_set.update(frozenset(self._spacetime_code.inactivation_sets[stabilizer_ranking[si_iter]].nonzero()[1]))            



        return self._spacetime_code.final_correction(correction)


@dataclass
class BPOSDHybridCorrect():
    _bpd_final_round : None
    _bpd : None
    _spacetime_code : None
    _checks : None
    _rounds : int 

    def __init__(self, code : qldpc.QuantumCodeChecks, rounds : int, bp_osd_options : Dict, priors : Tuple[float, float]):
        data_prior, measurement_prior = priors

        object.__setattr__(self, '_bpd_final_round', bposd_decoder(
            code.checks.z,
            error_rate = data_prior,
            **bp_osd_options
        ))

        object.__setattr__(self, '_rounds', rounds)
        object.__setattr__(self, '_checks', code.checks.z)
        object.__setattr__(self, '_spacetime_code', SpacetimeCode(self._checks, rounds))

        channel_prior = np.zeros(self._spacetime_code.spacetime_check_matrix.shape[1])
        self._spacetime_code.data_bits(channel_prior)[:] = data_prior
        self._spacetime_code.measurement_bits(channel_prior)[:] = measurement_prior
    
        object.__setattr__(self, '_bpd', bp_decoder(
            self._spacetime_code.spacetime_check_matrix,
            channel_probs = channel_prior,
            **bp_osd_options))

    def readout_correction(self, history : Callable[[int], np.array], data_readout : np.array) -> np.array:
        syndrome = self._spacetime_code.syndrome_from_history(history, data_readout)
        correction = self._bpd.decode(syndrome)
        final_round_bp_correction = self._spacetime_code.final_correction(correction)
        
        # Correct the transverse readout
        data_readout = (final_round_bp_correction + data_readout)%2

        # Final round correction on read out data
        syndrome = (self._checks @ data_readout)%2
        final_correction = self._bpd_final_round.decode(syndrome)
        return (final_correction + final_round_bp_correction)%2

@dataclass
class BPDetectorCorrect():
    _bpd : None
    _detector_spacetime_code : None

    def __init__(self, detector_error_model : stim.DetectorErrorModel, bp_osd_options : Dict):
        detector_spacetime_code = DetectorSpacetimeCode(detector_error_model)
        object.__setattr__(self, '_detector_spacetime_code', detector_spacetime_code)
        
        bpd = bp_decoder(
            self._detector_spacetime_code.fault_check_matrix,
            channel_probs = self._detector_spacetime_code.fault_priors,
            **bp_osd_options)
        object.__setattr__(self, '_bpd', bpd)

    def readout_correction(self, detector_string : np.array) -> np.array:
        '''Return the corrected logical observable measurements'''

        num_detectors = self._detector_spacetime_code.fault_check_matrix.shape[0]
        syndrome = detector_string[:num_detectors]
        logicals = detector_string[num_detectors:]
        fault_set = self._bpd.decode(syndrome)
        logicals = (logicals + self._detector_spacetime_code.fault_map @ fault_set)%2
        return logicals
        

def run_simulation(samples, code, meas_prior, data_prior, noise_model, noise_model_args, bp_osd_options, rounds, decoder_mode):
    
    checks = code.checks
    logicals = code.logicals

    # X / Z syndrome extraction circuit timesteps
    x_steps = max(np.max(checks.x.sum(axis=0)), np.max(checks.x.sum(axis=1)))
    z_steps = max(np.max(checks.z.sum(axis=0)), np.max(checks.z.sum(axis=1)))
    
    # Make this return a class
    # Add X/Z syndrome extraction circuit depth
    storage_sim = qldpc.build_storage_simulation(rounds, noise_model(**noise_model_args), code, use_x_logicals = False)

    meas_prior = meas_prior(x_steps, z_steps)
    data_prior = data_prior(x_steps, z_steps)

    
    circuit_string = '\n'.join(list(storage_sim.circuit))
    circuit = stim.Circuit(circuit_string)

    error_model = circuit.detector_error_model()

    detectors = False
    # Add correct prior here 1/2 - (1-2p)^n/2
    # Return measurment/data bits from spacetime code
    if decoder_mode == 'bposd':
        decoder = BPOSDCorrect(code, rounds, bp_osd_options, (data_prior, meas_prior))
    elif decoder_mode == 'bposd_single_shot':
        decoder = BPOSDCorrectSingleShot(code, rounds, bp_osd_options, (data_prior, meas_prior))
    elif decoder_mode == 'bposd_hybrid':
        decoder = BPOSDHybridCorrect(code, rounds, bp_osd_options, (data_prior, meas_prior))
    elif decoder_mode == 'sibpd':
        decoder = SIBPDCorrection(code, rounds, bp_osd_options, (data_prior, meas_prior))
    elif decoder_mode == 'bpd_detector':
        decoder = BPDetectorCorrect(error_model, bp_osd_options)
        detectors = True
    else:
        raise RuntimeError('Unknown decoder operation mode')


    if detectors:
        sampler = circuit.compile_detector_sampler()
        batch = sampler.sample(samples, append_observables=True)
    else:
        sampler = circuit.compile_sampler()
        batch = sampler.sample(samples)

    logical_values = []
    for i in range(samples):
        batch01 = np.where(batch[i], 1, 0)
        if detectors:
            corrected_logicals = decoder.readout_correction(batch01)
            logical_values.append(np.any(corrected_logicals != 0))
        else:
            data_readout = storage_sim.data_view(batch01)
            history = partial(storage_sim.measurement_view, get_x_checks=False, measurement_vector=batch01)
            data_readout = (data_readout + decoder.readout_correction(history, data_readout))%2
            logical_values.append(np.any(GF2(logicals.z) @ GF2(data_readout) != 0))
    return logical_values


def add_bposd_args(parser):
    '''Add arguments associated with BP+OSD to an ArgumentParser'''
    parser.add_argument('--bp_max_iter', type=lambda x: int(x) if x is not None else None, help='Maximum number of iterations for BP. Default is the number of qubits in the code', default=None)
    parser.add_argument('--bposd_bp_method', choices=['ps', 'ms', 'msl', 'psl'], help='BP method (product-sum, min-sum, min-sum log, product-sum log)', default='ps')
    parser.add_argument('--bposd_ms_scaling_factor', type=float, help='Min sum scaling factor. Use variable scaling factor method if 0', default=0)
    parser.add_argument('--bposd_osd_method', choices=['osd_e', 'osd_cs', 'osd0'], help='OSD method', default='osd_cs')
    parser.add_argument('--si_cutoff', type=int, help='Maximum number of stabilizer inactivation iterations', default = 10)
    parser.add_argument('--bposd_osd_order', type=int, help='OSD search depth', default=7)

def unpack_bposd_args(parsed_args, code):
    '''Convert the BP+OSD command line arguments to the arguments passed to the BP+OSD decoder'''
    return {
        'max_iter':parsed_args.bp_max_iter if parsed_args.bp_max_iter is not None else code.checks.num_qubits,
        'bp_method':parsed_args.bposd_bp_method,
        'ms_scaling_factor':parsed_args.bposd_ms_scaling_factor,
        'osd_method':parsed_args.bposd_osd_method, #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
        'osd_order':parsed_args.bposd_osd_order,
        'si_cutoff':parsed_args.si_cutoff,
    }

def load_code(args):
    '''Load a code and its logicals from the command line arguments'''
    with args.code.open() as code_file:
        code = qldpc.read_quantum_code(code_file, validate_stabilizer_code=True)
    return code
