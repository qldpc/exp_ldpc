import qldpc
from argparse import ArgumentParser
from pathlib import Path, PosixPath
import sys
import numpy as np
from ldpc import bp_decoder
from qldpc import SpacetimeCode
from phi_distribution import *
import time

def decode_code(check_matrix, syndrome, prior, passin):
    # Decoder goes here
    if passin:
        bp = bp_decoder(check_matrix, channel_probs=prior)
    else:
        bp = bp_decoder(check_matrix)

    correction = bp.decode(syndrome)
    return correction

def calc_prob_odd_error(ps):
    accumulator = 0
    for i in range(len(ps)):
        accumulator = accumulator + ps[i] - 2 * accumulator * ps[i]
    return accumulator

def run_simulation(samples, passin, code_path, d, p, **kwargs):

    with code_path.open() as code_file:
        code = qldpc.read_quantum_code(code_file)

    # doing arbitrary 10 rounds of stabilizer measurements for now
    spacetime_code = SpacetimeCode(code.checks.z, 10)

    results = []
    # t00 = time.time()

    for _ in range(samples):
        # I hope 0 is no error, and 1 is error
        error = np.zeros(spacetime_code.spacetime_check_matrix.shape[1], dtype=np.uint32)
        prior = np.zeros(spacetime_code.spacetime_check_matrix.shape[1], dtype=np.float64)

        # t_i = time.time()
        for i in range(spacetime_code.spacetime_check_matrix.shape[1]):
            tau = round(3*np.sqrt(code.num_qubits)*d/10)
            # t1 = time.time()
            p_phi_taus = np.array(sample_phi(tau, d, p))
            # t11 = time.time()
            # x_is = np.array([np.random.choice([0, 1], p=[1 - prob, prob]) for prob in p_phi_taus])
            x_is = np.where(1, 0, np.random.uniform(size=len(p_phi_taus)) <= p_phi_taus)
            # t12 = time.time()
            # print('sampling phi took', t11 - t1)
            # print('generating x_i took', t12 - t11)
            error_i = np.sum(x_is) % 2
            error[i] = error_i

            # t2 = time.time()
            prior[i] = calc_prob_odd_error(p_phi_taus)
            # t3 = time.time()
            # print('calculating prob of odd error of', tau, 'probabilities: ', t3 - t2)

        error = np.array(error)
        # t_f = time.time()

        # print('whole process of generating error took: ', t_f - t_i)
        # if not adding in idle measurements
        # llr_data, prior = sample_phi(spacetime_code.spacetime_check_matrix.shape[1], d, p)
        # error = np.array([np.random.choice([0, 1], p=[1-prob, prob]) for prob in prior])

        syndrome = (spacetime_code.spacetime_check_matrix @ error)%2 #syndrome in 01, all0s = all good
        correction = decode_code(spacetime_code.spacetime_check_matrix, syndrome, prior, passin) # a bunch of 0s when everything is correct
        # t_f2 = time.time()
        # print('running the decoder took: ', t_f2 - t_f)
        correction_single = spacetime_code.final_correction(correction)
        corrected_error = (spacetime_code.final_correction(error) + correction_single)%2
        logical_values = (code.logicals.z @ corrected_error)%2
        failure = np.any(logical_values != 0)
        results.append(failure)

    percentage_false = sum(value == False for value in results) / len(results) * 100

    t_final = time.time()
    # print('total time for 1 sample: ', t_final - t00)
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

    p = 0.018
    d = 15
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
