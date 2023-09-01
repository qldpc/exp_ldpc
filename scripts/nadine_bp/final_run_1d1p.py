from bp_decoder import *
import numpy as np
import time

def run_bp_decoder(p, d, syndmeas, samples=1000000):
    code_path = PosixPath('new_lifted_product_code.qecc')  # vars(args)['code_path']

    # make sure phi distribution is there, if not, then run it, and then save it
    run_phi_distribution(d, p, syndmeas, definitely_run=True, samples=samples)
    actually_save_json(d, p, syndmeas)

    # passin = True
    # print(f'p = {p}, d = {d}, pass_input={passin}')
    # start_time = time.time()
    # result = run_simulation(samples, passin, code_path, d, p, r)
    # end_time = time.time()
    #
    # print(f'after {samples} runs: {result}')
    # print(f'it took {(end_time - start_time)/60} min\n')
    #
    # with open(f'bp_decoder_output/d_{d}_p_{p}_r_{r}.txt', "w") as file:
    #     file.write(f'after {samples} runs: {np.average(result)}')

if __name__ == '__main__':

    # start = 0.0009
    # end = 0.0011865234
    # d = 7
    # r = 10
    # syndmeas = 69  # 6
    # # ============================ # #
    start = 0.011155 #0.00015
    end = 0.002
    d = 5
    syndmeas = 487 #682

    ps = [0.003] #[0.0054]
    for i in range(1):  # 10 points
        p = ps[i]
        run_bp_decoder(p, d, syndmeas, 1000000)

    # # ============================== #
    # start = 0.00829
    # end = 0.0125
    # d = 5
    # r = 10
    # syndmeas = 49
    #
    # samples = [30000, 30000, 25000, 2500, 2000, 2000, 200, 150, 100, 50]
    # ps = np.geomspace(start, end, 10)
    # for i in range(6, 9): # 10 points
    #     p = ps[i]
    #     run_bp_decoder(p, d, r, syndmeas, samples[i])
