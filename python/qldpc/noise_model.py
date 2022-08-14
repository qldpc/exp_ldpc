from collections import deque
from typing import Iterable, Deque
from itertools import chain
from .storage_sim import circuit_ticks
import re

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


def depolarizing_noise(p : float, pm : float, data_qubit_indices : Iterable[int], _ancilla_qubit_indices : Iterable[int], circuit : Iterable[str]) -> Deque[str]:
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
      
    return noisy_circuit

def timestep_depolarizing_noise(p : float, pm : float, data_qubit_indices : Iterable[int], ancilla_qubit_indices : Iterable[int], circuit : Iterable[str]) -> Deque[str]:
    '''Apply depolarizing noise to *all* qubits at rate p in each timestep. Also flip measurements with probability pm'''
    noisy_circuit = deque()
    qubit_indices = list(chain(data_qubit_indices , ancilla_qubit_indices))
    for timestep in circuit_ticks(circuit):
        # Add depolarizing noise
        noisy_circuit.append(f'DEPOLARIZE1({p}) {" ".join(str(i) for i in qubit_indices)}')
        noisy_circuit.extend(rewrite_measurement_noise(pm, line) for line in timestep)
      
    return noisy_circuit
