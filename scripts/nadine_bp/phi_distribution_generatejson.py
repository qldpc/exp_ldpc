import json
import os
import ast
from collections import defaultdict
import re
import subprocess

def actually_save_json(d, p, syndmeas):
    direc = "/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/phi_distr_rust_newhalf"

    data = calc_phidistr(d, p, syndmeas)
    print(data)
    if len(data) != 0:
        filename = f'{direc}/d_{d}_p_{p}_syndmeas_{syndmeas}_faultymeas.json'
        with open(filename, "w") as file:
            json.dump(data, file)

def calc_phidistr(d, p, syndmeas, faulty=True):
    merged_data_nofailure = extract_data(d, p, syndmeas, failure=False, faulty=faulty)
    merged_data_failure = extract_data(d, p, syndmeas, failure=True, faulty=faulty)

    pr_log_failure = calc_problogfailure(merged_data_nofailure, merged_data_failure)

    # make dict
    distr = defaultdict(int)
    for key in merged_data_nofailure:
        sum_values = merged_data_nofailure[key] + merged_data_failure[key]
        distr[pr_log_failure[key]] += sum_values

    return distr


def extract_data(d, p, syndmeas, failure=True, faulty=False):
    # Directory where the files are located
    if faulty:
        directories = [
            '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_tempeh_computer/UFD/phi_data_newhalf_1e7']

    merged_data_failure = defaultdict(int)

    if failure:
        string = f"d{d}_failure_p_{p}_syndromemeas_{syndmeas}"
    else:
        string = f"d{d}_nofailure_p_{p}_syndromemeas_{syndmeas}"

    total_nums = 0
    # Iterate over the files in the directory
    for directory in directories:
        for file_name in os.listdir(directory):
            if file_name.startswith(string):
                print(directory, file_name)
                match = re.search(r'num(\d+)', file_name)

                if match:
                    number_after_num = match.group(1)
                    total_nums += int(number_after_num)

                file_path = os.path.join(directory, file_name)

                # Read the file contents
                with open(file_path, "r") as file:
                    file_contents = file.read()

                # Parse the dictionary from the file contents
                data_dict = ast.literal_eval(file_contents) #use json.load next time

                # Merge the dictionary into the merged_data dictionary
                for key, value in data_dict.items(): # use one line next time
                    merged_data_failure.setdefault(key, 0)
                    merged_data_failure[key] += value

    return merged_data_failure


def calc_problogfailure(merged_data_nofailure, merged_data_failure):
    pr_log_failure = {}
    for key in merged_data_nofailure:
        sum_values = merged_data_nofailure[key] + merged_data_failure[key]
        pr_log_failure[key] = merged_data_failure[key] / sum_values
    return pr_log_failure

# ps = [0.009, 0.009525600656, 0.01008189643, 0.01067067991, 0.0112938484, 0.01195340997, 0.01265148998, 0.01339033792, 0.01417233463, 0.015]
# ps = [0.006, 0.006480358433, 0.006999174237, 0.007559526299, 0.008164740001, 0.008818406954, 0.009524406312, 0.01028692779, 0.01111049655, 0.012]
# for p in [0.008, 0.008216086292]:
#     actually_save_json(7, p, 95)


def run_phi_distribution(d, p, syndmeas, definitely_run=False, samples=1000000):
    filename = f'/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/phi_distr_rust_newhalf/d_{d}_p_{p}_syndmeas_{syndmeas}_faultymeas.json'

    # TODO: assumes once its run, its generated... maybe not necessarily true
    if definitely_run or not os.path.exists(filename):
        print('did not find the filename path soooo running it again', filename)
        current_directory = os.getcwd()
        os.chdir('/Users/nadinemeister/Documents/Harvard/Physics/Caltech/chris_tempeh_computer/UFD/target/release')
        # run it
        command = f'./ufd {round(1.5*p, 10)} {p} 0 0 0 {d} {syndmeas} {samples} false UnionFind true Depolarizing RotatedSurfaceCode'

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            print("Command executed successfully!")
            print("Output:", stdout.decode("utf-8"))
        else:
            print("Command failed!")
            print("Error:", stderr.decode("utf-8"))

        # generate file
        actually_save_json(d, p, syndmeas)

        os.chdir(current_directory)

ps = [0.005, 0.006] #[0.007, 0.00761774, 0.00829, 0.009082158612021489, 0.01, 0.010900795329560196]
for p in ps:
    print('savinggg')
    actually_save_json(7, p, 69)

# ps = [0.00829, 0.009082158612021489, 0.01, 0.010900795329560196]
# for p in ps:
#     actually_save_json(5, p, 49)
