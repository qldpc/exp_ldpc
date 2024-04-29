from collections import defaultdict
import random
import numpy as np
import time
import json
from functools import lru_cache
from numba import njit,jit
import numba
from numba import njit, float64, intp, bool_

def get_phidistr(d, p, num_meas, num_samples='1e6'):
    filename = f'/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/phi_distr_rust_newhalf/d_{d}_p_{p}_syndmeas_{num_meas}_faultymeas.json'
    # filename = f'/Users/nadinemeister/Dropbox/My Mac (Nadine’s MacBook Pro)/Documents/Harvard/Physics/Caltech/chris_qldpc/exp_ldpc/scripts/nadine_bp/phi_new_{num_samples}/d_{d}_p_{p}_syndmeas_{num_meas}_faultymeas_{num_samples}.json'

    with open(filename, "r") as data_file:
        phi_distr = json.load(data_file)

    # print(phi_distr)

    keys = np.array([float(x) for x in iter(phi_distr.keys())])
    frequencies = np.array(list(phi_distr.values()), dtype=np.float64)

    # Normalize the frequencies to ensure they sum up to 1
    total_frequency = np.sum(frequencies)
    normalized_frequencies = frequencies / total_frequency

    return keys, normalized_frequencies

def sample_phi(rng, size, phi_distr):
    q_values, phi_freq = phi_distr

    # Sample from the distribution
    sample_q = rng.choice(q_values, p=phi_freq, size=size)
    return sample_q
