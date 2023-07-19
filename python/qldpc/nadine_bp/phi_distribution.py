from collections import defaultdict
import random
import numpy as np
import time
import json

def load_phidistr(d, p, faulty=True):
    filename = f'phi_distr/d_{d}_p_{p}_faultymeas.json'

    with open(filename, "r") as file:
        loaded_data = json.load(file)

    return loaded_data

def sample_phi(num_samples, d, p):
    phi_distr = load_phidistr(d, p)

    keys = list(map(float, phi_distr.keys()))
    frequencies = list(phi_distr.values())

    # Normalize the frequencies to ensure they sum up to 1
    total_frequency = sum(frequencies)
    normalized_frequencies = [f / total_frequency for f in frequencies]

    # Sample from the distribution
    sample_cphi = random.choices(keys, weights=normalized_frequencies, k=num_samples)
    sample_q = np.exp(sample_cphi) / (1 + np.exp(sample_cphi))
    return sample_q
