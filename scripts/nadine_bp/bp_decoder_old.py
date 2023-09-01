import qldpc
from argparse import ArgumentParser
from pathlib import Path, PosixPath
import sys
import numpy as np
from ldpc import bp_decoder
from qldpc import SpacetimeCode
from phi_distribution import *
from time import time
from numba import njit

def decode_code(check_matrix, syndrome, prior, passin, iterations):
    # Decoder goes here
    if passin:
        bp = bp_decoder(check_matrix, channel_probs=prior, max_iter=iterations)
    else:
        bp = bp_decoder(check_matrix, max_iter=iterations)

    correction = bp.decode(syndrome)
    return correction

@njit(inline='always')
def calc_prob_odd_error(ps):
    accumulator = 0.0
    for i in range(len(ps)):
        accumulator = accumulator + ps[i] - 2 * accumulator * ps[i]
    return accumulator

@njit
def sample_error_prior_kernel(p_phi_tau):
    error = np.zeros(p_phi_tau.shape[0], dtype=np.uint32)
    prior = np.zeros(p_phi_tau.shape[0], dtype=np.float64)

    for i in range(p_phi_tau.shape[0]):
        mask = np.random.uniform(0.0, 1.0, p_phi_tau.shape[1]) <= p_phi_tau[i,:]
        x_is = np.where(mask, 1, 0)
        error_i = np.sum(x_is) % 2
        error[i] = error_i
        prior[i] = calc_prob_odd_error(p_phi_tau[i,:])
    return error, prior

def sample_error_prior(rng, size, tau, phi_distr):
    # I hope 0 is no error, and 1 is error
    p_phi_tau = sample_phi(rng, (size,tau), phi_distr)
    error, prior = sample_error_prior_kernel(rng, p_phi_tau)
        
    return error, prior

def run_simulation(samples, passin, code_path, d, p, r, **kwargs):

    with code_path.open() as code_file:
        code = qldpc.read_quantum_code(code_file)

    # doing arbitrary 10 rounds of stabilizer measurements for now
    spacetime_code = SpacetimeCode(code.checks.z, 10)

    rng = np.random.default_rng()
    
    results = []

    tau = int(np.ceil(3*np.sqrt(code.num_qubits)*d/r))
    phi_distr = get_phidistr(d, p, d * tau)

    for _ in range(samples):
        error, prior = sample_error_prior(rng, spacetime_code.spacetime_check_matrix.shape[1], tau, phi_distr)
        error = np.array(error)

        # if not adding in idle measurements
        # llr_data, prior = sample_phi(spacetime_code.spacetime_check_matrix.shape[1], d, p)
        # error = np.array([np.random.choice([0, 1], p=[1-prob, prob]) for prob in prior])

        syndrome = (spacetime_code.spacetime_check_matrix @ error)%2 #syndrome in 01, all0s = all good
        correction = decode_code(spacetime_code.spacetime_check_matrix, syndrome, prior, passin, iterations=code.num_qubits) # a bunch of 0s when everything is correct

        correction_single = spacetime_code.final_correction(correction)
        corrected_error = (spacetime_code.final_correction(error) + correction_single)%2
        logical_values = (code.logicals.z @ corrected_error)%2
        failure = np.any(logical_values != 0)
        results.append(failure)

    percentage_false = sum(value == False for value in results) / len(results) * 100

    print('percentage logically corrected', percentage_false)
    return percentage_false

def main():
    # parser = ArgumentParser(description='Sample and correct bit flip noise')
    # parser.add_argument('code_path', type=Path)
    # parser.add_argument('--samples', type=int, required=True, help='Number of samples to take')
    # parser.add_argument('--p_phys', type=float, required=True, help='physical error rate to sample at')

    # args = parser.parse_args(sys.argv[1:])
    # result = run_simulation(**vars(args))

    # stupider way to run this:
    code_path = PosixPath('hgp_1.qecc') #vars(args)['code_path']
    samples = 2 #vars(args)['samples']

    p = 0.01
    d = 5
    synd_meas = 45

    passin = True
    print(f'p = {p}, d = {d}, pass_input={passin}')
    results = []
    count = 0
    while True:
        count += 1
        result = run_simulation(samples, passin, code_path, d, p, 100)
        results.append(result)
        print(f'after {samples*count} runs: {np.average(results)}')

if __name__ == '__main__':
    main()
