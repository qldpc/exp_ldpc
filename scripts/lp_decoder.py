import qldpc
from ldpc import bp_decoder
from argparse import ArgumentParser
from pathlib import Path
import sys
import numpy as np
import cvxpy as cp
from itertools import permutations, chain, repeat, product
from belief_prop import BeliefPropagation

def enum_even_odd(n, even):
    return chain((permutations([True]*w+[False]*(n-w))) for w in range(0 if even else 1, n, 2))

def test_enum_even_odd():
    for (n,even) in product([3,4,5,6,7], [True,False]):
        num_entries = 0
        for entry in enum_even_odd(n, even):
            num_entries += 1
            is_even = sum(int(x) for x in entry)%2 == 0
            assert is_even == even
        assert num_entries == 2**(n-1)

def decode_code_ldpclib(code,syndrome, p_phys):
    ldpc_bpd = bp_decoder(code.checks.z, error_rate=p_phys, max_iter=200, bp_method='psl')
    return ldpc_bpd.decode(syndrome)
        
def decode_code_bp(code, syndrome, p_phys):
    # Sign on LLR prior????
    llr_prior = np.ones(code.num_qubits)*np.log((1-p_phys)/p_phys)
    bpd = BeliefPropagation(code.checks.z)
    return bpd.decode(syndrome, llr_prior, 20, break_converged=True)

def run_simulation(samples, p_phys, code_path, **kwargs):
    with code_path.open() as code_file:
        code = qldpc.read_quantum_code(code_file)
    checks = code.checks
    # Be super careful mixing this with multiprocessing!
    rng = np.random.default_rng()
    results = []
    
    for _ in range(samples):
        error = rng.choice([0,1], size=code.num_qubits, p=[1-p_phys, p_phys]).astype(np.uint32)
        syndrome = (code.checks.z @ error)%2
        correction = decode_code_bp(code, syndrome, p_phys)
        
        corrected_error = (error + correction)%2
        logical_values = (code.logicals.z @ corrected_error)%2
        failure = np.any(logical_values != 0)
        results.append(failure)
    return results

def main():
    parser = ArgumentParser(description='Sample and correct bit flip noise')
    parser.add_argument('code_path', type=Path)
    parser.add_argument('--samples', type=int, required=True, help='Number of samples to take')
    parser.add_argument('--p_phys', type=float, required=True, help='physical error rate to sample at')
    
    
    args = parser.parse_args(sys.argv[1:])


    result = run_simulation(**vars(args))
    print(f'Failures: {sum(result)}/{len(result)}')

if __name__ == '__main__':
    main()
