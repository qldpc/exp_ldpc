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
    # start = 0.011155 #0.00015
    # end = 0.002
    # d = 5
    # syndmeas = 487 #682
    # # ============================ # #
    # ds = [5, 7, 9, 11, 13]
    # ps = [0.08]
    # syndmeas = 49
    #
    # ps = [0.00829, 0.009082158612021489, 0.01, 0.010900795329560196] #[0.0054]
    # # while True:
    # for d in ds:
    #     for i in range(len(ps)):  # 10 points
    #         p = ps[i]
    #         run_bp_decoder(p, d, 0, 1000000)
    #
    # ps = [0.008677044133439568, 0.009506187105356915, 0.0104145596002901, ]

    # ============================== #
    start = 0.0036
    end = 0.005
    d = 5
    r = 10
    syndmeas = 49

    # samples = [30000, 30000, 25000, 2500, 2000, 2000, 200, 150, 100, 50]
    ps = np.geomspace(start, end, 5)
    p = 0.08
    # while True:
    #     for d in [9, 11, 13]: # 10 points
    #         run_bp_decoder(p, d, 0, 1000000)

    # ps = [0.014] #[0.01345, 0.014]#[0.011155, 0.011409732468538701, 0.01194243090232193, 0.0125] #[0.010696103757250688, 0.011440663558587229, 0.01223705244744459, 0.013088878266078581, 0.01360532]
    # for i in range(len(ps)):  # 10 points
    #     p = ps[i]
    run_bp_decoder(0.01091, 5, 49, 1000000)
    # run_bp_decoder(0.0051, 5, 487, 1000000)
    # run_bp_decoder(0.0053, 5, 487, 1000000)