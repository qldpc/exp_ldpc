from .homological_product_codes import random_test_hpg
from .qecc_util import QuantumCodeChecks, make_check_matrix



def d3_rotated_surface_code() -> QuantumCodeChecks:
    x_checks = [
        [0,1,3,4],
        [2,5],
        [3,6],
        [4,5,7,8],
    ]
    z_checks = [
        [0,1],
        [1,2,4,5],
        [3,4,6,7],
        [7,8],
    ]
    qubit_count = 9
    return (make_check_matrix(x_checks, qubit_count), make_check_matrix(z_checks, qubit_count), qubit_count)