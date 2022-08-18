from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
from itertools import chain
from argparse import ArgumentParser
from pathlib import Path
import sys
import pandas as pd
import re 
from typing import Tuple
from datetime import datetime

from _experiment import run_simulation, add_bposd_args, unpack_bposd_args, load_code

def p_sweep(samples, p_values, **kwargs):
    num_procs = cpu_count()

    max_samples = samples//num_procs
    residual = samples%num_procs if samples%num_procs != 0 else max_samples

    sample_distribution = [max_samples]*(num_procs-1) + [residual]
    assert np.sum(sample_distribution) == samples

    data = []
    with Pool(num_procs) as pool:
        for p_ph in p_values:
            time_start = datetime.now()

            run_shots = partial(run_simulation, p_ph=p_ph, **kwargs)
            logical_values = list(chain(*pool.map(run_shots, sample_distribution)))
            runtime = (datetime.now() - time_start).total_seconds()

            data_point = {'p_ph':p_ph, 'failures':sum(logical_values), 'samples':len(logical_values), 
                'walltime':runtime, **kwargs, **(kwargs['bp_osd_options'])}

            # Don't attach the code to the output
            del data_point['code']
            # Already expanded this
            del data_point['bp_osd_options']
            data.append(data_point)
    return pd.DataFrame.from_records(data)

# match_float = re.compile('^[+-]?\d+(?:\.\d+)?(?:[eE][+-]\d+)?$')
sweep_spec_re = re.compile('^\s*[(](.+),(.+),(.+)[)]\s*$')

def parse_sweep_spec(x : str) -> Tuple[float, float, int]:
    '''Parse strings of the form (a, b, c) where a,b : float, c : int, a<=b, and c > 0'''
    result = sweep_spec_re.match(x)
    if result is None:
        raise RuntimeError('Unable to parse sweep specification, expecting (a, b, c) where a,b : float, c : int, a<=b, and c > 0. Ex: (0.3, 1e3, 10)')
    lower, upper, points = result.group(1,2,3)
    lower, upper, points = (float(lower), float(upper), int(points))
    spec = (lower, upper, points)
    if points <= 0 or lower > upper:
        raise RuntimeError('Number of points non-positive or lower bound exceeded upper bound')
    return spec

if __name__ == '__main__':
    parser = ArgumentParser(description='Perform a parallelized sweep in the physical error rate for the given quantum code under BP+OSD')
    parser.add_argument('checks', type=Path)
    parser.add_argument('logicals', type=Path)
    parser.add_argument('--samples', type=int, help='Number of samples to take')
    parser.add_argument('--p_sweep', type=parse_sweep_spec, help='Specify lower and upper bounds of the sweep + number of points in the form (lower, upper, points)')
    parser.add_argument('--rounds', type=int, help='Number of rounds of syndrome extraction', default=1)
    parser.add_argument('--single_shot', type=bool, help='Operate decoder in single shot mode', default=False)
    parser.add_argument('--linspace', type=bool, help='Perform the sweep with linearly spaced points. The default is uniform spacing in log space', default=False)
    add_bposd_args(parser)

    args = parser.parse_args(sys.argv[1:])
    code = load_code(args)
    bp_osd_options = unpack_bposd_args(args, code)
    
    sweep = np.linspace(*args.p_sweep) if args.linspace else np.geomspace(*args.p_sweep)

    result = p_sweep(samples=args.samples, code=code, rounds=args.rounds, 
        p_values=sweep, single_shot=args.single_shot, bp_osd_options=bp_osd_options)
    
    result.to_csv(sys.stdout)
