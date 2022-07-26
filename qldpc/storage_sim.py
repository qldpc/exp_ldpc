from multiprocessing.sharedctypes import Value
import networkx as nx
from networkx.algorithms import bipartite
import scipy.sparse as sparse
from typing import Callable, Iterable, Tuple, Dict, List, Deque
from .edge_coloring import edge_color_bipartite
from .qecc_util import num_rows, QuantumCodeChecks, QuantumCodeLogicals
import itertools
from collections import deque
import re
import numpy as np

MeasurementOrder = Tuple[int, Dict[int, int]]

def order_measurements(checks : QuantumCodeChecks) -> Tuple[int, MeasurementOrder, MeasurementOrder]:
    '''Returns an ordering of measuremnts for X and Z checks. For now we schedule X and Z checks separately'''
    def build_meas_order_basis(checks):
        tanner_graph = bipartite.from_biadjacency_matrix(checks)
        check_nodes, data_nodes = bipartite.sets(tanner_graph, top_nodes=set(range(num_rows(checks))))
        # We depend on the indexing of the graph to be check_nodes then data_nodes
        data_node_offset = len(check_nodes)
        for i in range(data_node_offset):
            assert i in check_nodes

        scheduling = edge_color_bipartite(tanner_graph)
        meas_order = []
        for round in range(len(scheduling)):
            target_round = dict()
            for edge in scheduling[round]:
                assert edge[0] in check_nodes
                target_round[edge[0]] = edge[1] - data_node_offset
            meas_order.append(target_round)
        return (len(data_nodes), len(check_nodes), meas_order)


    (x_data_nodes, x_check_nodes, xorder) = build_meas_order_basis(checks[0])
    (z_data_nodes, z_check_nodes, zorder) = build_meas_order_basis(checks[1])
    assert x_data_nodes == z_data_nodes
    return (x_data_nodes, (x_check_nodes, xorder), (z_check_nodes, zorder) )

def build_perfect_circuit(checks : QuantumCodeChecks) -> Tuple[List[int], List[int], List[int], List[str]]:
    '''Syndrome extraction circuit to measure X checks then Z checks'''
    (num_data_qubits, (x_check_count, x_check_schedule), (z_check_count, z_check_schedule)) = order_measurements(checks)

    x_check_ancillas = np.array(list(range(num_data_qubits, num_data_qubits+x_check_count)))
    x_check_ancilla_str = ' '.join(str(v) for v in x_check_ancillas)

    z_check_ancillas = np.array(list(range(num_data_qubits+x_check_count, num_data_qubits+x_check_count+z_check_count)))
    z_check_ancilla_str = ' '.join(str(v) for v in z_check_ancillas)
    
    circuit = []
    # Init X check ancillas
    circuit.append(f'RX {x_check_ancilla_str}')
    circuit.append('TICK') # --------
    # X check circuit
    for round in x_check_schedule:
        circuit.extend(f'CX {x_check_ancillas[check]} {target}' for (check, target) in round.items())
        circuit.append('TICK') # --------
    # Measure
    circuit.append(f'MRX {x_check_ancilla_str}')
    
    # Init Z check ancillas in parallel with X check measurements
    circuit.append(f'RX {z_check_ancilla_str}')
    circuit.append('TICK') # --------

    for round in z_check_schedule:
        circuit.extend(f'CZ {z_check_ancillas[check]} {target}' for (check, target) in round.items())
        circuit.append('TICK') # --------

    circuit.append(f'MRX {z_check_ancilla_str}')

    # Leave off the final tick so we can interleave this element
    # circuit.append('TICK')  # --------
    return (list(range(num_data_qubits)), x_check_ancillas, z_check_ancillas, circuit)

measurement_gates = ['M', 'MZ', 'MX', 'MY', 'MPP', 'MR', 'MRZ', 'MRX', 'MRY']
measurement_line_pattern = re.compile(f'^(?:\\s*)({"|".join(measurement_gates)})((?:\\s*\\d+\\s*)+)$')

def rewrite_measurement_noise(p : float, circuit_line : str) -> str:
    '''Rewrite all measurements to contain noise with parameter p'''
    search_result = measurement_line_pattern.search(circuit_line)
    if search_result is None:
        return circuit_line
    else:
        (meas_type, targets) = search_result.group(1,2)
        return f'{meas_type}({p}){targets}'

def circuit_ticks(circuit : Iterable[str]) -> Deque[Deque[str]]:
    '''Returns a list of subcircuits separated by a TICK'''
    subcircuits = deque()
    for k, g in itertools.groupby(circuit, lambda x: x.strip().upper() == 'TICK'):
        if k != True:
            subcircuits.append(deque(g))
    return subcircuits

def depolarizing_noise_model(p : float, pm : float, data_qubit_indices : Iterable[int], _ancilla_qubit_indices : Iterable[int], circuit : Iterable[str]) -> Deque[str]:
    '''Apply depolarizing noise to data qubits with rate p in any timestep where measurements take place. Also flip measurements with probability pm'''
    noisy_circuit = deque()
    for timestep in circuit_ticks(circuit):
        try:
            # Look for a measurement in this timestep
            next(filter(lambda line: measurement_line_pattern.search(line) is not None, timestep))
            # Add depolarizing noise
            noisy_circuit.append(f'DEPOLARIZE1({p}) {" ".join(str(i) for i in data_qubit_indices)}')
            noisy_circuit.extend(rewrite_measurement_noise(pm, line) for line in timestep)
        except StopIteration:
            # Identity if no measurement
            noisy_circuit.extend(timestep)
        noisy_circuit.append('TICK')
    # Remove the last TICK
    noisy_circuit.pop()
    return noisy_circuit

noise_channels = (
    'CORRELATED_ERROR',
    'DEPOLARIZE1',
    'DEPOLARIZE2',
    'ELSE_CORRELATED_ERROR',
    'PAULI_CHANNEL_1',
    'PAULI_CHANNEL_2',
    'X_ERROR',
    'Y_ERROR',
    'Z_ERROR',
)

def unique_targets(circuit : str):
    '''Ensure that a qubit only appears once per timestep. TODO: check only the targets of gates'''
    
    def try_int(x):
        try:
            return int(x)
        except ValueError:
            return None

    def unique_targets_timestep(step : str):
        targets = [x for x in map(try_int, step.split()) if x is not None]
        unique_targets = frozenset(targets)
        assert len(targets) == len(unique_targets)

    # A qubit index appears a second time in the noise annotation so we need to remove them to check that all gate targets are unique
    def discard_noise(circuit : str):
        return '\n'.join(s for s in circuit.split('\n') if not s.startswith(noise_channels))

    for timestep, timestep_circuit in enumerate(circuit.split('TICK')):
        unique_targets_timestep(discard_noise(timestep_circuit))

# TODO: We will rewrite the perfect circuit to insert the appropriate fault locations noise model
def build_storage_simulation(rounds : int, noise_model : Callable[[str], str], checks : QuantumCodeChecks, use_x_logicals = None) -> Tuple[str, Callable[[int, bool, list], list], Callable[[list], list]]:
    '''Construct a simulation where a logical 0 is prepared stored for rounds number of QEC cycles then transversally read out
    use_x_logicals: prepare a |+> and read out in the X basis'''
    if use_x_logicals is None:
        use_x_logicals = False

    reset_meas_basis = 'X' if use_x_logicals else 'Z'

    (data_qubit_indices, x_check_indices, z_check_indices, syndrome_extraction_circuit) = build_perfect_circuit(checks)

    x_check_count = len(x_check_indices)
    z_check_count = len(z_check_indices)

    circuit = []
    # Prepare logical zero
    # We could optimize this by directly measuring the checks but it's a small cost
    circuit.append(f'R{reset_meas_basis} {" ".join(str(i) for i in data_qubit_indices)}')
    circuit.append('TICK')

    # Do QEC
    for _ in range(rounds):
        circuit.extend(syndrome_extraction_circuit)

        
    # Read out data qubits
    circuit.append(f'M{reset_meas_basis} {" ".join(str(i) for i in data_qubit_indices)}')

    # Rewrite circuit with noise model
    circuit = noise_model(data_qubit_indices, x_check_indices+z_check_indices, circuit)

    def meas_result(round_index, get_x_checks, measurement_vector, *_, x_check_count=x_check_count, z_check_count=z_check_count):
        meas_round_offset = (x_check_count + z_check_count) * round_index
        check_offset = meas_round_offset + (0 if get_x_checks else x_check_count)
        check_count = x_check_count if get_x_checks  else z_check_count

        return measurement_vector[check_offset : check_offset + check_count]

    def data_result(measurement_vector, *_, x_check_count=x_check_count, z_check_count=z_check_count, rounds=rounds, num_data_qubits = len(data_qubit_indices)):
        offset = (x_check_count + z_check_count)*rounds
        return measurement_vector[offset : offset*num_data_qubits]

    unique_targets('\n'.join(circuit))
    return (circuit, meas_result, data_result)


def test_noise_rewrite():
    circuit = [
        'RX 0 1 2',
        'TICK',
        'CZ 0 1',
        'TICK',
        'MX 0 2',
        'TICK',
        'TICK',
        'MX 0',
    ]

    rewritten_circuit = depolarizing_noise_model(0.1, 0.2, [1], [0,2], circuit)

    # Golden test for now
    print(rewritten_circuit)
    assert rewritten_circuit[4] == 'MX(0.2) 0 2'
    assert rewritten_circuit[5] == 'DEPOLARIZE1(0.1) 1'
    assert rewritten_circuit[7] == 'MX(0.2) 0'
    assert rewritten_circuit[8] == 'DEPOLARIZE1(0.1) 1'
