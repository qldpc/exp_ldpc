import networkx as nx
from networkx.algorithms import bipartite
import scipy.sparse as sparse
from typing import Callable, Iterable, Tuple, Dict, List
from .edge_coloring import edge_color_bipartite
from .qecc_util import num_rows, QuantumCodeChecks, QuantumCodeLogicals

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
        return (len(check_nodes), meas_order)
    
    return (len(data_nodes), build_meas_order_basis(checks[0]), build_meas_order_basis(checks[1]))

def build_perfect_circuit(checks : QuantumCodeChecks) -> Tuple[int, int, int, Tuple[List[int], List[int], List[int]], List[str]]:
    '''Syndrome extraction circuit to measure X checks then Z checks'''
    (num_data_qubits, (x_check_count, x_check_schedule), (z_check_count, z_check_schedule)) = order_measurements(checks)

    x_check_ancillas = list(range(num_data_qubits, num_data_qubits+x_check_count))
    x_check_ancilla_str = ' '.join(str(v) for v in x_check_ancillas)

    z_check_ancillas = list(range(num_data_qubits+x_check_count, num_data_qubits+x_check_count+z_check_count))
    z_check_ancilla_str = ' '.join(str(v) for v in z_check_ancillas)
    
    circuit = []
    # Init X check ancillas
    circuit.append(f'RX {x_check_ancilla_str}')
    circuit.append('TICK') # --------
    # X check circuit
    for round in x_check_schedule:
        circuit.extend(f'CX {check} {target}' for (check, target) in round.items())
        circuit.append('TICK') # --------
    # Measure
    circuit.append(f'MX {x_check_ancilla_str}')
    
    # Init Z check ancillas in parallel with X check measurements
    circuit.append(f'RX {z_check_ancilla_str}')
    circuit.append('TICK') # --------

    for round in z_check_schedule:
        circuit.extend(f'CZ {check} {target}' for (check, target) in round.items())
        circuit.append('TICK') # --------

    circuit.append(f'MX {z_check_ancilla_str}')

    # Leave off the final tick so we can interleave this element
    # circuit.append('TICK')  # --------

    return (num_data_qubits, x_check_count, z_check_count, (list(range(num_data_qubits)), x_check_ancillas, z_check_ancillas), circuit)

def rewrite_measurement_noise(p : float, circuit_line : str) -> str:
    measurement_gates = ['M', 'MZ', 'MX', 'MY', 'MPP', 'MR', 'MRZ', 'MRX', 'MRY']
    (left_partition, measurement, right_partition) = circuit_line.partition(measurement_gates)
    assert(left_partition.isspace or len(left_partition) == 0)
    return f'{measurement}({p}){right_partition}'

def depolarizing_noise_model(p : float, pm : float, data_qubit_indices : Iterable[int], ancilla_qubit_indices : Iterable[int], circuit : Iterable[str]) -> List[str]:
    noisy_circuit = [rewrite_measurement_noise(pm, line) for line in circuit]
    noisy_circuit.append(f'DEPOLARIZE1({p}) {" ".join(data_qubit_indices)}')
    noisy_circuit.append(f'DEPOLARIZE1({p}) {" ".join(ancilla_qubit_indices)}')
    return noisy_circuit


# TODO: We will rewrite the perfect circuit to insert the appropriate fault locations noise model
def build_storage_simulation(rounds : int, noise_model : Callable[[str], str], logicals : QuantumCodeLogicals, checks : QuantumCodeChecks, use_x_logicals = None) -> Tuple[str, Callable[[int, bool, list], list], Callable[[list], list]]:
    if use_x_logicals is None:
        use_x_logicals = False

    reset_meas_basis = 'X' if use_x_logicals else 'Z'

    (num_data_qubits, x_check_count, z_check_count, syndrome_extraction_circuit) = build_perfect_circuit(checks)

    circuit = []
    # Prepare logical zero
    # We could optimize this by directly measuring the checks but it's a small cost
    circuit.append(f'R{reset_meas_basis} {" ".join(range(num_data_qubits))}')
    circuit.extend(syndrome_extraction_circuit)

    # Do noisy QEC
    noisy_syndrome_extraction_circuit = noise_model(syndrome_extraction_circuit)
    for _ in range(rounds):
        circuit.extend(noisy_syndrome_extraction_circuit)
    
    # Perfect QEC round
    circuit.extend(syndrome_extraction_circuit)

    # Read out data qubits
    circuit.append(f'M{reset_meas_basis} {" ".join(range(num_data_qubits))}')

    def meas_result(round_index, get_x_checks, measurement_vector):
        meas_round_offset = (x_check_count + z_check_count) * round_index
        check_offset = meas_round_offset + (0 if get_x_checks else x_check_count)
        check_count = x_check_count if get_x_checks  else z_check_count

        return measurement_vector[check_offset : check_offset + check_count]

    def data_result(measurement_vector):
        offset = (x_check_count + z_check_count)*rounds
        return measurement_vector[offset : offset*num_data_qubits]

    return (circuit, meas_result, data_result)
