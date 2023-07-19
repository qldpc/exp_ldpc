from collections import defaultdict
import ast
import os
import random
import statsmodels.api as sm
import numpy as np

def extract_data(d, p, failure=True, faulty=False):
    # Directory where the files are located
    if faulty:
        directories = ['/Users/nadinemeister/PycharmProjects/qsurface/tests/data_actualfaultymeas/',
                     '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_cluster/',
                     '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_cluster_transferred/',
                     '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_cluster_higher_transferred/',
                     '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_cluster_higher_transferred2/',
                     '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_cluster_d25/',
                     '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/d51_faulty/',
                     '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_cluster_morehighertransfferred']
    else:
        directories = ['/Users/nadinemeister/PycharmProjects/qsurface/tests/data',
                       '/Users/nadinemeister/PycharmProjects/qsurface/tests/data_LARGE',
                       '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_cluster5199/',
                       '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/d61_101_faultyFALSE/',
                       '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_clusterfaultyFalse/',
                       '/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/data_clusterPerfect/']
    merged_data_failure = defaultdict(int)

    if failure:
        string = f"d{d}_failure_p_{p}_"
    else:
        string = f"d{d}_nofailure_p_{p}_"

    # Iterate over the files in the directory
    for directory in directories:
        for file_name in os.listdir(directory):
            if file_name.startswith(string):
                file_path = os.path.join(directory, file_name)

                with open(file_path, "r") as file:
                    file_contents = file.read()

                data_dict = ast.literal_eval(file_contents)

                for key, value in data_dict.items():
                    merged_data_failure.setdefault(key, 0)
                    merged_data_failure[key] += value

    return merged_data_failure


def calc_problogfailure(d, p, faulty=False):
    merged_data_nofailure = extract_data(d, p, failure=False, faulty=faulty)
    merged_data_failure = extract_data(d, p, failure=True, faulty=faulty)

    pr_log_failure = {}
    for key in merged_data_nofailure:
        sum_values = merged_data_nofailure[key] + merged_data_failure[key]
        pr_log_failure[key] = merged_data_failure[key] / sum_values
    return pr_log_failure

def calc_phidistr(d, p, faulty=True):
    merged_data_nofailure = extract_data(d, p, failure=False, faulty=faulty)
    merged_data_failure = extract_data(d, p, failure=True, faulty=faulty)

    # get slope
    pr_log_failure = calc_problogfailure(d, p, faulty=faulty)
    sorted_data = sorted(pr_log_failure.items(), key=lambda item: float(item[0]))
    og_x = [float(item[0]) for item in sorted_data]
    y = np.array([item[1] for item in sorted_data])
    loggedy = np.log(y / (1 - y))
    valid_indices = np.isfinite(loggedy)
    x_valid = np.array(og_x)[valid_indices]
    y_valid = np.array(loggedy)[valid_indices]
    x = x_valid
    y = y_valid
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    results = model.fit()
    intercept, slope = results.params

    # make dict
    distr = {}
    for key in merged_data_nofailure:
        sum_values = merged_data_nofailure[key] + merged_data_failure[key]
        distr[float(key) * slope] = sum_values

    return distr

def sample_phi(num_samples, d, p):
    if d == 7 and p == 0.02: # to make sure it's slowed down by going through all the files
        phi_distr = {-4.245923254463293: 1018006,
                     -10.614808136158233: 1214966,
                     -8.491846508926587: 1949122,
                     -2.1229616272316467: 504618,
                     -6.368884881694941: 1694989,
                     -0.0: 133790,
                     -12.737769763389881: 152383,
                     -14.860731390621527: 2126}
    elif d == 7 and p == 0.026:
        phi_distr = {-3.5398031395840888: 27055,
                     -0.0: 7098,
                     -5.309704709376133: 29088,
                     -7.0796062791681775: 21399,
                     -1.7699015697920444: 17651,
                     -8.84950784896022: 7268,
                     -10.619409418752266: 441}
    else:
        phi_distr = calc_phidistr(d, p)

    keys = list(map(float, phi_distr.keys()))
    frequencies = list(phi_distr.values())

    # Normalize the frequencies to ensure they sum up to 1
    total_frequency = sum(frequencies)
    normalized_frequencies = [f / total_frequency for f in frequencies]

    # Sample from the distribution
    sample_cphi = random.choices(keys, weights=normalized_frequencies, k=num_samples)
    sample_q = np.exp(sample_cphi) / (1 + np.exp(sample_cphi))
    return sample_q
