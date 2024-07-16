from networkx.algorithms import bipartite
from typing import Callable, Iterable, Tuple, Dict, List, Deque

from .noise_model import NoiseRewriter
from .edge_coloring import edge_color_bipartite
from .qecc_util import num_rows, QuantumCode, QuantumCodeChecks, NoiseRewriter, CircuitTargets, StorageSim
from collections import deque
import numpy as np

MeasurementOrder = Tuple[int, Dict[int, int]]

def order_measurements(code : QuantumCode) -> Tuple[int, MeasurementOrder, MeasurementOrder]:
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


    (x_data_nodes, x_check_nodes, xorder) = build_meas_order_basis(code.checks.x)
    (z_data_nodes, z_check_nodes, zorder) = build_meas_order_basis(code.checks.z)
    assert x_data_nodes == z_data_nodes
    return (x_data_nodes, (x_check_nodes, xorder), (z_check_nodes, zorder) )

def build_perfect_circuit(code : QuantumCode) -> Tuple[CircuitTargets, List[str]]:
    '''Syndrome extraction circuit to measure X checks then Z checks'''
    
    (num_data_qubits, (x_check_count, x_check_schedule), (z_check_count, z_check_schedule)) = order_measurements(code)

    x_check_ancillas = np.array(list(range(num_data_qubits, num_data_qubits+x_check_count)))
    x_check_ancilla_str = ' '.join(str(v) for v in x_check_ancillas)

    z_check_ancillas = np.array(list(range(num_data_qubits+x_check_count, num_data_qubits+x_check_count+z_check_count)))
    z_check_ancilla_str = ' '.join(str(v) for v in z_check_ancillas)
    
    circuit = []
    # Init X check ancillas
    circuit.append(f'RX {x_check_ancilla_str}')
    circuit.append('TICK') # --------

    if x_check_count > 0:
    # X check circuit
        for round in x_check_schedule:
            circuit.extend(f'CX {x_check_ancillas[check]} {target}' for (check, target) in round.items())
            circuit.append('TICK') # --------
        # Measure
        circuit.append(f'MRX {x_check_ancilla_str}')
    
    # Init Z check ancillas in parallel with X check measurements
    circuit.append(f'RX {z_check_ancilla_str}')
    circuit.append('TICK') # --------

    if z_check_count > 0:
        for round in z_check_schedule:
            circuit.extend(f'CZ {z_check_ancillas[check]} {target}' for (check, target) in round.items())
            circuit.append('TICK') # --------

        circuit.append(f'MRX {z_check_ancilla_str}')

    # Leave off the final tick so we can interleave this element
    # circuit.append('TICK')  # --------
    return (CircuitTargets(list(range(num_data_qubits)), list(x_check_ancillas), list(z_check_ancillas)), circuit)

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

def _check_unique_targets(circuit : str):
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
        return '\n'.join(s for s in circuit.split('\n') if not s.startswith(noise_channels) and not s.startswith(('DETECTOR',  'OBSERVABLE')))

    for timestep, timestep_circuit in enumerate(circuit.split('TICK')):
        unique_targets_timestep(discard_noise(timestep_circuit))

def build_storage_simulation(rounds : int, noise_model : NoiseRewriter, code : QuantumCode, use_x_logicals = None) -> StorageSim:
    '''Construct a simulation where a logical 0 is prepared stored for rounds number of QEC cycles then transversally read out
    use_x_logicals: prepare a |+> and read out in the X basis.
    This is starting to become a pretty leaky abstraction and should probably be replaced with a new design at some point.'''
    
    if use_x_logicals is None:
        use_x_logicals = False

    checks = code.checks

    reset_meas_basis = 'X' if use_x_logicals else 'Z'

    targets, syndrome_extraction_circuit = build_perfect_circuit(code)

    # Ensure the indices are contiguous since we will assume this later
    assert all(targets.x_checks[i+1] == targets.x_checks[i] + 1 for i in range(len(targets.x_checks)-1))
    assert all(targets.z_checks[i+1] == targets.z_checks[i] + 1 for i in range(len(targets.z_checks)-1))

    if len(targets.z_checks) > 0 and len(targets.x_checks) > 0:
        assert targets.z_checks[0] == targets.x_checks[-1]+1
    
    x_check_count = len(targets.x_checks)
    z_check_count = len(targets.z_checks)

    circuit = []
    # ===== Initialize State ====
    circuit.append(f'R{reset_meas_basis} {" ".join(str(i) for i in targets.data)}')
    circuit.append('TICK')
    
    # ===== Repeated Measurement Rounds ====
    measurements_per_round = x_check_count + z_check_count
    
    if rounds > 0:
        # First round
        circuit.extend(syndrome_extraction_circuit)
        # Since we start in a product state, only one measurment type is deterministic
        circuit.extend(f'DETECTOR(0, {i}) rec[{i-measurements_per_round}]' for i in
                       (range(0,x_check_count) if use_x_logicals else range(x_check_count, measurements_per_round)))
        
        if rounds > 1:
            # Steady state rounds
            circuit.append(f'TICK')
            circuit.append(f'REPEAT {rounds-1} {{')
            circuit.extend(syndrome_extraction_circuit)
            circuit.append('SHIFT_COORDS(1, 0)')
            # Detector annotations
            circuit.extend(f'DETECTOR(0, {i}) rec[{i-measurements_per_round}] rec[{i-2*measurements_per_round}]'
                                      for i in range(measurements_per_round))
            circuit.append('TICK')
            circuit.append('}')

    # ===== Final Round =====
    # Read out data qubits
    circuit.append(f'M{reset_meas_basis} {" ".join(str(i) for i in targets.data)}')

    # Detectors for final round
    records = lambda support: ' '.join(f'rec[{v-len(targets.data)}]' for v in support)
    if use_x_logicals:
        circuit.extend(f'DETECTOR(1, {i}) '
            + (f'rec[{i-len(targets.data)-measurements_per_round}] ' if rounds > 0 else '')   # previous round measurement
            + records(checks.x[[i],:].nonzero()[1])                                      # current round syndrome
            for i in range(checks.x.shape[0]))
        circuit.extend(f'OBSERVABLE_INCLUDE({i}) '
            + records(np.nonzero(code.logicals.x[[i],:])[0])
            for i in range(code.logicals.x.shape[0]))
    else:
        circuit.extend(f'DETECTOR(1, {i}) '
            + (f'rec[{i-len(targets.data)-measurements_per_round+x_check_count}] ' if rounds > 0 else '')   # previous round measurement
            + records(checks.z[[i],:].nonzero()[1])                                      # current round syndrome
            for i in range(checks.z.shape[0]))
        circuit.extend(f'OBSERVABLE_INCLUDE({i}) '
            + records(np.nonzero(code.logicals.z[[i],:])[0])
            for i in range(code.logicals.z.shape[0]))

    # Rewrite circuit with noise model
    circuit = noise_model.rewrite(targets, circuit)

    def meas_result(round_index, get_x_checks, measurement_vector, *_, x_check_count=x_check_count, z_check_count=z_check_count):
        meas_round_offset = (x_check_count + z_check_count) * round_index
        check_offset = meas_round_offset + (0 if get_x_checks else x_check_count)
        check_count = x_check_count if get_x_checks  else z_check_count

        return measurement_vector[check_offset : check_offset + check_count]

    def data_result(measurement_vector, *_, x_check_count=x_check_count, z_check_count=z_check_count, rounds=rounds, num_data_qubits = len(targets.data)):
        offset = (x_check_count + z_check_count)*rounds
        return measurement_vector[offset : offset+num_data_qubits]

    _check_unique_targets('\n'.join(circuit))
    return StorageSim(circuit, meas_result, data_result)
