from qldpc import *
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
    parser.add_argument('--rounds', type=int, help='Number of rounds of syndrome extraction', default=1)
    parser.add_argument('--seed', type=lambda x: int(x) if x is not None else None, help='PRNG seed', default=None)
    parser.add_argument('--save_code', type=Path, help='File path to save the code to')
    parser.add_argument('--save_circuit', type=Path, help='File path to save the syndrome extraction circuit to')

    args = parser.parse_args(sys.argv[1:])

    if args.save_code is not None and args.save_code.exists():
        print('Code save destination already exists')
        exit(-1)

    if args.save_circuit is not None and args.save_circuit.exists():
        print('Circuit save destination already exists')
        exit(-1)

    (checks, _) = biregular_hpg(args.nv, args.dv, args.dc, seed=args.seed)

    id_noise_model = lambda a, b, x: x
    circuit, _, _ = build_storage_simulation(args.rounds, id_noise_model, checks, use_x_logicals = False)

    if args.save_code is not None:
        with args.save_code.open('w') as code_file:
            write_check_generators(code_file, checks)
    else:
        write_check_generators(sys.stdout, checks)

    if args.save_circuit is not None:
        with args.save_circuit.open('w') as circuit_file:
            circuit_file.write('\n'.join(circuit))