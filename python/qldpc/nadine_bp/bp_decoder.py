import qldpc
from argparse import ArgumentParser
from pathlib import Path, PosixPath
import sys
import numpy as np
from ldpc import bp_decoder
from qldpc import SpacetimeCode
from phi_distribution import *

def decode_code(check_matrix, syndrome, prior, passin):
    # Decoder goes here
    if passin:
        bp = bp_decoder(check_matrix, channel_probs=prior)
    else:
        bp = bp_decoder(check_matrix)

    correction = bp.decode(syndrome)
    return correction

def calc_prob_odd_error(ps):
    if len(ps) == 2:
        p1 = ps[0]
        p2 = ps[1]
        return p1 * (1-p2) + (1-p1) * p2
    else:
        p1 = calc_prob_odd_error([ps[0], ps[1]])
        return calc_prob_odd_error(np.append(p1, ps[2:]))

def run_simulation(samples, passin, code_path, d, p, **kwargs):

    with code_path.open() as code_file:
        code = qldpc.read_quantum_code(code_file)

    # doing arbitrary 10 rounds of stabilizer measurements for now
    spacetime_code = SpacetimeCode(code.checks.z, 10)

    results = []

    for _ in range(samples):
        # I hope 0 is no error, and 1 is error
        error = []
        prior = []

        for _ in range(spacetime_code.spacetime_check_matrix.shape[1]):
            tau = round(3*np.sqrt(code.num_qubits)*d/10)
            p_phi_taus = sample_phi(tau, d, p)
            x_is = np.array([np.random.choice([0, 1], p=[1 - prob, prob]) for prob in p_phi_taus])
            error_i = np.sum(x_is) % 2
            error.append(error_i)

            prior_i = calc_prob_odd_error(p_phi_taus)
            prior.append(prior_i)

        # if not adding in idle measurements
        # llr_data, prior = sample_phi(spacetime_code.spacetime_check_matrix.shape[1], d, p)
        # error = np.array([np.random.choice([0, 1], p=[1-prob, prob]) for prob in prior])

        syndrome = (spacetime_code.spacetime_check_matrix @ error)%2 #syndrome in 01, all0s = all good
        correction = decode_code(spacetime_code.spacetime_check_matrix, syndrome, prior, passin) # a bunch of 0s when everything is correct

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
    samples = 1 #vars(args)['samples']

    p = 0.026
    d = 7
    passin = True
    print(f'p = {p}, d = {d}, pass_input={passin}')
    results = []
    count = 0
    while True:
        count += 1
        result = run_simulation(samples, passin, code_path, d, p)
        results.append(result)
        print(f'after {samples*count} runs: {np.average(results)}')

if __name__ == '__main__':
    main()
