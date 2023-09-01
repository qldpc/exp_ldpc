import os
import json
from phi_distribution_generatejson import *
from bp_decoder import run_simulation
from pathlib import PosixPath
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

def get_distribution_and_runbp_decoder(d, p, syndmeas, r, passin):
    # get phi distribution
    run_phi_distribution(d, p, syndmeas, samples=1000000)

    # run bp
    # if it's not automatically 0% or 100%, then run it for 1000 samples and save that value
    code_path = PosixPath('/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/new_lifted_product_code.qecc') #vars(args)['code_path']
    samples = 50

    print(f'p = {p}, d = {d}, pass_input={passin}')
    perc_succ = run_simulation(samples, passin, code_path, d, p, r) #100 is the r
    if perc_succ == 0:
        return 0
    else:
        print('not 0, so running 1000 bp_decoders')
        start = time.time()
        perc_succ = run_simulation(1000, passin, code_path, d, p, r)
        end = time.time()
        print(f'those took this long {(end-start)/60}')

    return perc_succ



def save_and_plot(d, p, perc_succ, syndmeas, df, passin):
    new_row = pd.DataFrame({'p': [p],
                            'prob_success': [perc_succ],
                            'num_samples': [1000],
                            'd': [d]})
    df = pd.concat([df, new_row], ignore_index=True)
    csv_filename = f'/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/perc_success_data/output_data_d_{d}_r_1_syndmeas_{syndmeas}_passin_{passin}.csv'
    df.to_csv(csv_filename, index=False)

    plt.figure()
    df = df.sort_values(by='p', ascending=True)

    plt.loglog(df.p, 1 - np.array(df.prob_success) / 100, 'o-', label='soft info')
    # plt.legend()
    plt.grid()
    plt.title(f'Simulations for Hierarchical code \n with d = {d} surface code \n')
    plt.ylabel('Hierarchical logical failure rate')
    plt.xlabel('Individual physical qubit error rate')
    plt.tight_layout()
    plt.savefig(f'/Users/nadinemeister/Documents/Harvard/Physics/Caltech/images/NewHierBP_d{d}_r_{r}_passin_{passin}.png')#, dpi=300)
    plt.close()
    return df

if __name__ == '__main__':
    # essentially binary search
    d = 7
    r = 1
    syndmeas = int(np.ceil(3 * np.sqrt(1054) * d / r))

    # 1. determine range to binary search on
    left = 0
    right = 0.02
    data = {
        'p': [left, right],
        'd': [d, d],
        'prob_success': [100, 0],
        'num_samples': [1000, 1000]
    }
    passin = True #hard info

    # Create a DataFrame
    df = pd.DataFrame(data)

    # next run
    while True:
        print('L, R', left, right)
        p = round((left + right) / 2, 10)
        perc_succ = get_distribution_and_runbp_decoder(d, p, syndmeas, r, passin=passin)

        # save data + plot it
        df = save_and_plot(d, p, perc_succ, syndmeas, df, passin=passin)

        # update p before running again
        if perc_succ == 100:
            print('in left', left, right)
            left = p
            print('after setting left in left', left, right)
        elif perc_succ == 0:
            print('in right', left, right)
            right = p
            print('after setting right in right', left, right)
        else:
            print('in else', left, right)
            right = (p + right) / 2
            print()

    # 2. run bp_decoder