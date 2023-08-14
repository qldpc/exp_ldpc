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
import numba
from numba import njit, float64, intp, bool_
from phi_distribution_generatejson import *

def decode_code(check_matrix, syndrome, prior, passin, iterations):
    # Decoder goes here
    if passin:
        bp = bp_decoder(check_matrix, channel_probs=prior, max_iter=iterations)
    else:
        bp = bp_decoder(check_matrix, max_iter=iterations)

    correction = bp.decode(syndrome)
    return correction

# @njit
def sample_error_prior(rng, size, phi_distr):
    # I hope 0 is no error, and 1 is error
    q_values, phi_freq = phi_distr

    # Sample from the distribution
    prior = rng.choice(q_values, p=phi_freq, size=size)
    error = np.random.uniform(0.0, 1.0, prior.shape[0]) <= prior

    return error, prior


def run_simulation(samples, passin, code_path, d, p, r, **kwargs):

    with code_path.open() as code_file:
        code = qldpc.read_quantum_code(code_file)

    # doing arbitrary 10 rounds of stabilizer measurements for now
    spacetime_code = SpacetimeCode(code.checks.z, 10)

    rng = np.random.default_rng()
    
    results = []

    tau = int(np.ceil(3*np.sqrt(code.num_qubits)*d/r))
    phi_distr = get_phidistr(d, p, tau)

    for _ in range(samples):
        error, prior = sample_error_prior(rng, spacetime_code.spacetime_check_matrix.shape[1], phi_distr)
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
    samples = 10 #vars(args)['samples']

    ps_7 = [0.009, 0.009525600656, 0.01008189643, 0.01067067991, 0.0112938484, 0.01195340997, 0.01265148998, 0.01339033792, 0.01417233463, 0.015]
    ps_5 = [0.006, 0.006480358433, 0.006999174237, 0.007559526299, 0.008164740001, 0.008818406954, 0.009524406312, 0.01028692779, 0.01111049655, 0.012]

    p = ps_5[8] #
    d = 5
    r = 10
    syndmeas = 68
    save_results = False

    actually_save_json(d, p, syndmeas)

    passin = True
    print(f'p = {p}, d = {d}, pass_input={passin}')
    results = []
    count = 0
    while True:
        count += 1
        result = run_simulation(samples, passin, code_path, d, p, r)
        results.append(result)
        print(f'after {samples*count} runs: {np.average(results)}')

        if save_results:
            with open(f'bp_decoder_output/d_{d}_p_{p}_r_{r}.txt', "w") as file:
                file.write(f'after {samples*count} runs: {np.average(results)}')


if __name__ == '__main__':
    main()
