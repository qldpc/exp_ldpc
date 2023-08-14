import json
import os
import ast
from collections import defaultdict
import re

def actually_save_json(d, p, syndmeas):
    direc = "/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/phi_distr_rust"

    data = calc_phidistr(d, p, syndmeas)
    if len(data) != 0:
        filename = f'{direc}/d_{d}_p_{p}_syndmeas_{syndmeas}_faultymeas.json'
        with open(filename, "w") as file:
            json.dump(data, file)


def calc_phidistr(d, p, syndmeas, faulty=True):
    merged_data_nofailure = extract_data(d, p, syndmeas, failure=False, faulty=faulty)
    merged_data_failure = extract_data(d, p, syndmeas, failure=True, faulty=faulty)

    # get slope
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
            '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_tempeh_computer/UFD/phi_data']

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
                data_dict = ast.literal_eval(file_contents)

                # Merge the dictionary into the merged_data dictionary
                for key, value in data_dict.items():
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



