import pandas as pd
import os
import subprocess

def run_ufd_surfcode(d, p, syndmeas):
    # run ./ufd without phi calculation
    # first do like 100 or so runs, then do 1e5 runs?
    os.chdir('/Users/nadinemeister/Documents/Harvard/Physics/Caltech/chris_tempeh_computer/UFD/target/release')
    command = f'./ufd {1.5 * p} {p} 0 0 0 {d} {syndmeas} 10000 false UnionFind true Depolarizing RotatedSurfaceCode'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == '__main__':
    d = 9 # will go up to 1e5 range

    # 1. determine range to binary search on
    left = 0
    right = 1

    data = {
        'p': [left, right],
        'd': [d, d],
        'prob_fail': [0, 1],
        'num_samples': [1000, 1000]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # next run
    while True:
        print('L, R', left, right)
        p = round((left + right) / 2, 10)
        perc_fail = run_ufd_surfcode(d, p)

        # save data + plot it
        df = save_and_plot(d, p, perc_fail, df)

        # update p before running again
        if perc_fail == 0:
            print('in left', left, right)
            left = p
            print('after setting left in left', left, right)
        elif perc_fail == 1:
            print('in right', left, right)
            right = p
            print('after setting right in right', left, right)
        else:
            print('in else', left, right)
            right = (p + right) / 2
            print()

