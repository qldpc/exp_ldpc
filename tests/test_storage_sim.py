from itertools import chain
from collections import deque

from qldpc import depolarizing_noise_model, build_storage_simulation
from qldpc.storage_sim import build_perfect_circuit
from qldpc.code_examples import random_test_hgp

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
    assert rewritten_circuit[4] == 'DEPOLARIZE1(0.1) 1'
    assert rewritten_circuit[5] == 'MX(0.2) 0 2'
    assert rewritten_circuit[8] == 'DEPOLARIZE1(0.1) 1'
    assert rewritten_circuit[9] == 'MX(0.2) 0'

def test_ancilla_targets():
    # Reconstruct the checks from the syndrome extraction circuit and verify they match the code
    checks, _ = random_test_hgp(compute_logicals=False)
    
    data_qubit_idx, x_ancilla_idx, z_ancilla_idx, circuit = build_perfect_circuit(checks)

    x_ancilla_idx = frozenset(x_ancilla_idx)
    measurement_order = deque(map(lambda x: int(x), chain(*[s.split()[1:] for s in circuit if s.startswith(('MX', 'MRX'))])))

    # Find all the targets of CZ/CX
    CX_targets = {i:set() for i in x_ancilla_idx}
    CZ_targets = {i:set() for i in z_ancilla_idx}
    for s in circuit:
        if s.startswith('CX'):
            _, control, target = s.split()
            CX_targets[int(control)].add(int(target))
        if s.startswith('CZ'):
            _, control, target = s.split()
            CZ_targets[int(control)].add(int(target))

    # Verify the CX/CZ targets match the check supports
    assert len(measurement_order) == checks.x.shape[0] + checks.z.shape[0]
    for (i,m) in enumerate(measurement_order):
        if m in x_ancilla_idx:
            assert CX_targets[m] == set(checks.x[i,:].nonzero()[1])
        else:
            assert CZ_targets[m] == set(checks.z[i-checks.x.shape[0],:].nonzero()[1])


def smoketest_storage_sim():
    noise_model = lambda *x: depolarizing_noise_model(p_ph, 0, *x)
    checks, _ = random_test_hgp(compute_logicals=False)
    build_storage_simulation(3, noise_model, checks.x, use_x_logicals = False)
