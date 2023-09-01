import pandas as pd
import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import glob
import re

def run_ufd_surfcode(d, p, syndmeas, samples=1000):
    # run ./ufd without phi calculation
    # first do like 100 or so runs, then do 1e5 runs?
    os.chdir('/Users/nadinemeister/Documents/Harvard/Physics/Caltech/chris_tempeh_computer/UFD/target/release')

    # first true is now true as in run just surface code data
    command = f'./ufd {round(1.5 * p, 10)} {p} 0 0 0 {d} {syndmeas} {samples} true UnionFind true Depolarizing RotatedSurfaceCode'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print("Command executed successfully!")
        print("Output:", stdout.decode("utf-8"))
    else:
        print("Command failed!")
        print("Error:", stderr.decode("utf-8"))


    # get the percent success
    directory = "/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_tempeh_computer/UFD/surf_code_data"
    data = []

    # Iterate through all .txt files in the specified directory
    for filename in glob.glob(os.path.join(directory, "d*_nofailure_p_*_syndromemeas_*_num_*.txt")):
        #     print("Processing file:", filename)

        # Use regular expressions to extract the required information from the filename
        match = re.match(r"d(\d+)_nofailure_p_(\d+\.\d+)_syndromemeas_(\d+)_num_(\d+)_", os.path.basename(filename))
        #     print(match, os.path.basename(filename))
        if match:
            d2 = int(match.group(1))
            p2 = float(match.group(2))
            syndmeas2 = int(match.group(3))
            num_samples = int(match.group(4))

            if d2 == d and p2 == p and syndmeas2 == syndmeas:
                # Read the content of the file
                with open(filename, 'r') as file:
                    content = file.read()
                    #             print(content)
                    # Extract the value after the colon
                    success_rate = float(content)

                # Append the extracted data to the list
                data.append((d, p, syndmeas, num_samples, int(success_rate * num_samples)))

    df_sc = pd.DataFrame(data, columns=['d_real', 'p', 'syndmeas', 'num_samples', 'num_fail'])
    agg_df = df_sc.groupby(['d_real', 'syndmeas', 'p']).agg({'num_samples': 'sum', 'num_fail': 'sum'}).reset_index()

    agg_df['p_fail'] = agg_df['num_fail']/agg_df['num_samples']
    agg_df['total_p_fail'] = 1 - (1 - agg_df['p_fail']) ** 140 #k = 140

    row_index = (agg_df['p'] == p) & (agg_df['d_real'] == d)
    value = agg_df.loc[row_index, 'total_p_fail'].item()


    return value


def save_and_plot(d_real, d_s, p, perc_fail, syndmeas, df):
    new_row = pd.DataFrame({'p': [p],
                            'd_s': [d_s],
                            'd_real': [d_real],
                            'prob_fail': [perc_fail],
                            'num_samples': [1000]})
    df = pd.concat([df, new_row], ignore_index=True)
    csv_filename = f'/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/perc_success_data/output_data_d_{d_s}_syndmeas_{syndmeas}.csv'
    df.to_csv(csv_filename, index=False)

    plt.figure()
    df = df.sort_values(by='p', ascending=True)

    plt.loglog(df.p, np.array(df.prob_fail), 'o-', label='soft info')
    # plt.legend()
    plt.grid()
    plt.title(f'Simulations for just surface codes (d = {d_real}) \n for d = {d_s} in Hierarchical Code \n')
    plt.ylabel('Surf Code logical failure rate')
    plt.xlabel('Individual physical qubit error rate')
    plt.tight_layout()
    plt.savefig(f'/Users/nadinemeister/Documents/Harvard/Physics/Caltech/images/JustSurfCode_ds{d_s}_dreal{d_real}.png')#, dpi=300)

    return df

if __name__ == '__main__':

    # 1. determine range to binary search on
    d_s = 5
    d_real = int(np.ceil(np.sqrt(1054/140)))*d_s
    left = 0.008
    right = 0.02
    syndmeas = 3*int(np.ceil(np.sqrt(1054)))*d_s

    data = {
        'p': [left, right],
        'd_s': [d_s, d_s],
        'd_real': [d_real, d_real],
        'prob_fail': [0, 1],
        'num_samples': [1000, 1000]
    }

    # Create a DataFrame
    df = pd.DataFrame(data)

    # next run
    while True:
        print('L, R', left, right)
        p = round((left + right) / 2, 10)
        perc_fail = run_ufd_surfcode(d_real, p, syndmeas, samples=10000)

        # save data + plot it
        df = save_and_plot(d_real, d_s, p, perc_fail, syndmeas, df)

        # update p before running again
        if perc_fail == 0:
            print('in left', left, right)
            left = p
            print('after setting left in left', left, right)
        elif perc_fail > 0.5:
            print('in right', left, right)
            right = p
            print('after setting right in right', left, right)
        else:
            print('in else', left, right)
            right = round((p + right) / 2, 10)
            print()

