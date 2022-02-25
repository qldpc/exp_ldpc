from multiprocessing.sharedctypes import Value
import networkx as nx
from networkx.algorithms import bipartite
import scipy.sparse as sparse
from typing import Callable, Iterable, Tuple, Dict, List
from .edge_coloring import edge_color_bipartite
from .qecc_util import num_rows, QuantumCodeChecks, QuantumCodeLogicals
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
    circuit.append(f'MX {x_check_ancilla_str}')
    
    # Init Z check ancillas in parallel with X check measurements
    circuit.append(f'RX {z_check_ancilla_str}')
    circuit.append('TICK') # --------

    for round in z_check_schedule:
        circuit.extend(f'CZ {z_check_ancillas[check]} {target}' for (check, target) in round.items())
        circuit.append('TICK') # --------

    circuit.append(f'MX {z_check_ancilla_str}')

    # Leave off the final tick so we can interleave this element
    # circuit.append('TICK')  # --------
    return (list(range(num_data_qubits)), x_check_ancillas, z_check_ancillas, circuit)

measurement_gates = ['M', 'MZ', 'MX', 'MY', 'MPP', 'MR', 'MRZ', 'MRX', 'MRY']
measurement_line_pattern = re.compile(f'^(?:\s*)({"|".join(measurement_gates)})((?:\s*\d+\s*)+)$')

def rewrite_measurement_noise(p : float, circuit_line : str) -> str:
    search_result = measurement_line_pattern.search(circuit_line)
    if search_result is None:
        return circuit_line
    else:
        (meas_type, targets) = search_result.group(1,2)
        return f'{meas_type}({p}){targets}'

def depolarizing_noise_model(p : float, pm : float, data_qubit_indices : Iterable[int], ancilla_qubit_indices : Iterable[int], circuit : Iterable[str]) -> List[str]:
    noisy_circuit = [rewrite_measurement_noise(pm, line) for line in circuit]
    noisy_circuit.append(f'DEPOLARIZE1({p}) {" ".join(str(i) for i in data_qubit_indices)}')
    noisy_circuit.append(f'DEPOLARIZE1({p}) {" ".join(str(i) for i in ancilla_qubit_indices)}')
    return noisy_circuit

def unique_targets(circuit : str):
    '''Ensure that a qubit only appears once per timestep'''
    def try_int(x):
        try:
            return int(x)
        except ValueError:
            return None

    def unique_targets_timestep(step : str):
        targets = [x for x in map(try_int, step.split()) if x is not None]
        unique_targets = frozenset(targets)
        assert len(targets) == len(unique_targets)

    for timestep, timestep_circuit in enumerate(circuit.split('TICK')):
        unique_targets_timestep(timestep_circuit)

# TODO: We will rewrite the perfect circuit to insert the appropriate fault locations noise model
def build_storage_simulation(rounds : int, noise_model : Callable[[str], str], checks : QuantumCodeChecks, use_x_logicals = None) -> Tuple[str, Callable[[int, bool, list], list], Callable[[list], list]]:
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
    circuit.extend(syndrome_extraction_circuit)

    # Do noisy QEC
    noisy_syndrome_extraction_circuit = noise_model(data_qubit_indices, x_check_indices+z_check_indices, syndrome_extraction_circuit)
    for _ in range(rounds):
        circuit.extend(noisy_syndrome_extraction_circuit)
    
    # Perfect QEC round
    circuit.extend(syndrome_extraction_circuit)

    # Read out data qubits
    circuit.append(f'M{reset_meas_basis} {" ".join(str(i) for i in data_qubit_indices)}')

    def meas_result(round_index, get_x_checks, measurement_vector, *_, x_check_count=x_check_count, z_check_count=z_check_count):
        meas_round_offset = (x_check_count + z_check_count) * round_index
        check_offset = meas_round_offset + (0 if get_x_checks else x_check_count)
        check_count = x_check_count if get_x_checks  else z_check_count

        return measurement_vector[check_offset : check_offset + check_count]

    def data_result(measurement_vector, *_, x_check_count=x_check_count, z_check_count=z_check_count, rounds=rounds, num_data_qubits = len(data_qubit_indices)):
        offset = (x_check_count + z_check_count)*(rounds+2)
        return measurement_vector[offset : offset*num_data_qubits]

    unique_targets('\n'.join(circuit))
    return (circuit, meas_result, data_result)

