from qldpc import biregular_hgp, build_storage_simulation, noise_model, write_quantum_code
from pathlib import Path
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a hypergraph product code and associated syndrome extraction circuit')
    parser.add_argument('dc', type=int, help='Classical code check degree in input to hypergraph product code')
    parser.add_argument('dv', type=int, help='Classical code vertex degree in input to hypergraph product code')
    parser.add_argument('nv', type=int, help='''
        Classical code data vertex count in input to hypergraph product code. 
        (n dv)/dc must be integer. The total number of qubits in the quantum code is (1+(dv/dc)^2) n^2, and the total number of checks is (2dv/dc) n^2.''')
    parser.add_argument('--girth_bound', type=int, help='Remove all cycles of length <= girth_bound in the classical tanner graph', default=None)
    parser.add_argument('--girth_bound_patience', type=int, help='Timeout for random swaps to achieve girth bound', default=100000)
    parser.add_argument('--rounds', type=int, help='Number of rounds of syndrome extraction', default=1)
    parser.add_argument('--seed', type=lambda x: int(x) if x is not None else None, help='PRNG seed', default=None)
    parser.add_argument('--save_code', type=Path, help='File path to save the code to')
    parser.add_argument('--save_circuit', type=Path, help='File path to save the syndrome extraction circuit to')
    parser.add_argument('--compute_logicals', type=bool, default=True, help='Optionally compute logical operators of the code, Warning: This has O(n^3) time complexity')

    args = parser.parse_args(sys.argv[1:])

    if args.save_code is not None and args.save_code.exists():
        print('Code save destination already exists')
        exit(-1)

    if args.save_circuit is not None and args.save_circuit.exists():
        print('Circuit save destination already exists')
        exit(-1)


    code = biregular_hgp(args.nv, args.dv, args.dc, seed=args.seed, compute_logicals=args.compute_logicals,
        girth_bound=args.girth_bound, girth_bound_patience=args.girth_bound_patience)

    storage_sim = build_storage_simulation(args.rounds, noise_model.trivial_noise(), code.checks, use_x_logicals = False)

    if args.save_code is not None:
        with args.save_code.open('w') as code_file:
            write_quantum_code(code_file, code)
    else:
        write_quantum_code(sys.stdout, code)

    if args.save_circuit is not None:
        with args.save_circuit.open('w') as circuit_file:
            circuit_file.write('\n'.join(storage_sim.circuit))
