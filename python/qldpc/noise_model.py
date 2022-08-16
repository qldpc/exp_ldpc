from collections import deque
import string
from typing import Iterable, Deque, List, Callable
from itertools import chain
from .qecc_util import NoiseRewriter, CircuitTargets
from functools import partial
import re

def trivial_noise() -> NoiseRewriter:
    '''Don't apply any noise'''
    return apply_noise_pred(lambda *_: False)

def apply_noise_pred(predicate : Callable[[CircuitTargets, Iterable[str]], bool],
    noise_before : Callable[[CircuitTargets], List[str]] = None,
    noise_after : Callable[[CircuitTargets], List[str]] = None,
    line_rewriter : Callable[[CircuitTargets, str], str] = None ) -> NoiseRewriter:
    '''Call predicate on each timestep. If it returns true then prepend noise_before, append noise_after, and rewrite the circuit line-by-line with noise_rewrite'''
    return NoiseRewriter(partial(_apply_noise_pred_impl,
        predicate=predicate,
        noise_after=noise_after,
        noise_before=noise_before,
        line_rewriter=line_rewriter))

def lift_line_pred(line_predicate : Callable[[CircuitTargets, str], bool]) -> Callable[[CircuitTargets, Iterable[str]], bool]:
    '''Lift a predicate that acts on a single lines to act on entire timesteps'''
    return lambda targets, subcircuit, *_, line_predicate=line_predicate: any(map(partial(line_predicate, targets), subcircuit))

def circuit_ticks(circuit : Iterable[str]) -> Deque[Deque[str]]:
    '''Returns a list of subcircuits separated by a TICK'''
    subcircuits = deque()
    subcircuits.append(deque())
    circuit_iter = iter(circuit)
    while True:
        try:
            line = next(circuit_iter)
            subcircuits[-1].append(line)
            if line.strip().upper() == 'TICK':
                subcircuits.append(deque())
        except StopIteration:
            break
    return subcircuits

_measurement_gates = ['M', 'MZ', 'MX', 'MY', 'MPP', 'MR', 'MRZ', 'MRX', 'MRY']
_measurement_line_pattern = re.compile(f'^(?:\\s*)({"|".join(_measurement_gates)})((?:\\s*\\d+\\s*)+)$')

def has_measurement_pred(*x):
    return lift_line_pred(lambda _, line: _measurement_line_pattern.search(line) is not None)(*x)

def depolarizing_noise(p : float, pm : float) -> NoiseRewriter:
    '''Apply depolarizing noise to data qubits with rate p in any timestep where measurements take place. Also flip measurements with probability pm'''
    
    noise_before = lambda targets, p=p: [f'DEPOLARIZE1({p}) {" ".join(str(i) for i in targets.data)}']
    return apply_noise_pred(
        predicate = has_measurement_pred, noise_before=noise_before,
        line_rewriter=partial(_rewrite_measurement_noise, p=pm))

def _rewrite_measurement_noise(_targets : CircuitTargets, circuit_line : str, p : float = None) -> str:
    '''Rewrite all measurements to contain noise with parameter p'''
    search_result = _measurement_line_pattern.search(circuit_line)
    if search_result is None:
        return circuit_line
    else:
        (meas_type, targets) = search_result.group(1,2)
        return f'{meas_type}({p}){targets}'

def _apply_noise_pred_impl(
    targets : CircuitTargets, circuit : Iterable[str], *_,
    predicate, noise_before, noise_after, line_rewriter) -> Iterable[str]:
    '''Implementation for apply_noise_pred'''

    if noise_before is None:
        noise_before = lambda *_: []
    if noise_after is None:
        noise_after = lambda *_: []
    if line_rewriter is None:
        line_rewriter = lambda _, x: x

    noisy_circuit = deque()
    for timestep in circuit_ticks(circuit):
        if predicate(targets, timestep):
            # Add noise
            noisy_circuit.extend(noise_before(targets))
            noisy_circuit.extend(line_rewriter(targets, line) for line in timestep)
            noisy_circuit.extend(noise_after(targets))
        else:
            noisy_circuit.extend(timestep)
      
    return noisy_circuit

