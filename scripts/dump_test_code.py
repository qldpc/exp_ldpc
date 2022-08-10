from qldpc import *
import numpy as np
import stim
import os

def list_to_arr_literal(x, newline=False):
    return '[' + f', {os.linesep if newline else ""}'.join(str(v) for v in x) + ']'

def dump_checks(checks):
    list_of_lists_checks = [list(checks[i, :].nonzero()[1]) for i in range(checks.shape[0])]
    print(list_to_arr_literal(list_of_lists_checks, newline=True))
        
def non_zeros(x):
    return [i for i,v in enumerate(x) if v != 0]

if __name__ == '__main__':
    (checks, logicals) = biregular_hpg(12, 5, 6, seed=670235982)
    print('x_checks\n')
    dump_checks(checks.x)
    print('z_checks\n')
    dump_checks(checks.z)
    print(os.linesep)
    print('x_logicals')
    print(list_to_arr_literal((non_zeros(x) for x in logicals[0]), newline=True))
