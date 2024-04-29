import qldpc
from pathlib import Path, PosixPath
from ldpc import bp_decoder, bposd_decoder
from phi_distribution import *
import os

import numpy as np
from qldpc import SpacetimeCode, SpacetimeCodeSingleShot, DetectorSpacetimeCode


def decode_code(check_matrix, syndrome, prior, passin, iterations, osd_method):
    # Decoder goes here
    # if passin:
    bp = bposd_decoder(check_matrix, channel_probs=prior, max_iter=iterations, osd_method=osd_method)#, bp_method='ms')
    # else:
    #     bp = bp_decoder(check_matrix, max_iter=iterations)

    correction = bp.decode(syndrome)
    # print(bp.iter)
    return correction

# @njit
def sample_error_prior(rng, size, phi_distr):
    # I hope 0 is no error, and 1 is error
    q_values, phi_freq = phi_distr

    # Sample from the distribution
    prior = rng.choice(q_values, p=phi_freq, size=size)
    np.random.seed(os.getpid())
    error = np.random.uniform(0.0, 1.0, size=prior.shape[0]) <= prior

    return error, prior

oT = 100
osd_method='osd_e'

def run_simulation(samples, passin, code_path, d, p, r, num_samples, **kwargs):

    with code_path.open() as code_file:
        code = qldpc.read_quantum_code(code_file)

    # doing arbitrary 10 rounds of stabilizer measurements for now
    spacetime_code = SpacetimeCode(code.checks.z, oT)

    rng = np.random.default_rng()
    
    results = []

    tau = int(np.ceil(3*np.sqrt(code.num_qubits)*d/r)) #num syndrome rounds
    phi_distr = get_phidistr(d, p, tau, num_samples)

    for _ in range(samples):
        # sample phi - number of qubits in the qLDPC code times
        error, prior = sample_error_prior(rng, spacetime_code.spacetime_check_matrix.shape[1], phi_distr)
        error = np.array(error)

        # if not adding in idle measurements
        # llr_data, prior = sample_phi(spacetime_code.spacetime_check_matrix.shape[1], d, p)
        # error = np.array([np.random.choice([0, 1], p=[1-prob, prob]) for prob in prior])

        syndrome = (spacetime_code.spacetime_check_matrix @ error)%2 #syndrome in 01, all0s = all good

        if passin == False:
            keys, normalized_freq = phi_distr
            prior_val = np.sum([keys[i]*normalized_freq[i] for i in range(len(keys))])
            prior = [prior_val] * len(prior)
        correction = decode_code(spacetime_code.spacetime_check_matrix, syndrome, prior, passin, iterations=code.num_qubits, osd_method=osd_method) # a bunch of 0s when everything is correct

        correction_single = spacetime_code.final_correction(correction)
        corrected_error = (spacetime_code.final_correction(error) + correction_single)%2
        logical_values = (code.logicals.z @ corrected_error)%2
        failure = np.any(logical_values != 0)
        results.append(failure)

    percentage_false = sum(value == False for value in results) / len(results) * 100

    print('percentage logically corrected', percentage_false)
    return percentage_false

def main():
    code_path = PosixPath('new_lifted_product_code.qecc') #vars(args)['code_path']
    samples = 10 #vars(args)['samples']
    # 5000, 5000, 5000, 1000, 100, 10
    save_results = False

    # d = 7
    # r = 1
    # syndmeas = 69

    #d=5, r=10
    ps = [0.007, 0.00761774, 0.00829, 0.009082158612021489, 0.01, 0.010900795329560196]
    # p = 0.0054

    # ps = [0.0096, 0.01013319, 0.010696103757250688, 0.011440663558587229, 0.01223705244744459, 0.013088878266078581, 0.01360532]

    #d=5,r=1
    # ps = [0.00312, 0.00338, 0.0036, 0.003908133374595783, 0.0042426406871192875, 0.004605779351596908, 0.005, 0.0051,
    #       0.0053, 0.015]

    d = 5
    r = 10
    # syndmeas = 49
    num_samples = '1e7'
    p = ps[2]

    # make sure phi distribution is there, if not, then run it, and then save it
    # run_phi_distribution(d, p, syndmeas, definitely_run=False, samples=1000000)
    # actually_save_json(d, p, syndmeas)

    passin = True
    print(f'p = {p}, d = {d}, pass_input={passin}')
    results = []
    count = 0
    while True:
        count += 1
        result = run_simulation(samples, passin, code_path, d, p, r, num_samples)
        results.append(result)
        print(f'after {samples*count} runs: {np.average(results)}')

        if save_results:
            with open(f'bposd_output/d_{d}_p_{p}_r_{r}_passin_{passin}_oT_{oT}_osd_{osd_method}.txt', "w") as file:
                file.write(f'after {samples*count} runs: {np.average(results)}')


if __name__ == '__main__':
    # profile.run('main()')
    main()
    # pr = cProfile.Profile()
    # pr.enable()
    # main()
    # pr.disable()
    # s = io.StringIO()
    # sortby = SortKey.TIME
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())