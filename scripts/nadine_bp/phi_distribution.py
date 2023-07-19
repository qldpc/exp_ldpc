from collections import defaultdict
import random
import numpy as np
import time
import json
from functools import lru_cache
from numba import njit,jit

def get_phidistr(d, p, faulty=True):
    filename = f'phi_distr/d_{d}_p_{p}_faultymeas.json'

    with open(filename, "r") as data_file:
        phi_distr = json.load(data_file)

    keys = np.array([float(x) for x in iter(phi_distr.keys())])
    frequencies = np.array(list(phi_distr.values()), dtype=np.float)

    # Normalize the frequencies to ensure they sum up to 1
    total_frequency = np.sum(frequencies)
    normalized_frequencies = frequencies / total_frequency

    return keys, normalized_frequencies

def sample_phi(rng, size, phi_distr):
    phi_values, phi_freq = phi_distr

    # Sample from the distribution
    sample_cphi = rng.choice(phi_values, p=phi_freq, size=size)
    sample_q = np.exp(sample_cphi) / (1 + np.exp(sample_cphi))
    return sample_q
