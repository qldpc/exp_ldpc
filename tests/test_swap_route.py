import numpy as np
import pytest
from qldpc import product_permutation_route, grid_permutation_route

def assert_disjoint_swaps(swap_schedule):
    # Check that all swaps in the same round are disjoint
    for round in swap_schedule:
        round_set = set()
        for swap_op in round:
            for target in swap_op:
                assert target not in round_set
                round_set.add(target)

def execute_swaps(a, swap_schedule):
    for round in swap_schedule:
        for swap_op in round:
            t = np.copy(a[swap_op[0][0],swap_op[0][1],:])
            a[swap_op[0][0],swap_op[0][1],:] = a[swap_op[1][0],swap_op[1][1],:]
            a[swap_op[1][0],swap_op[1][1],:] = t

def _random_permutation(G_size, H_size):
    permutation = np.array([(i,j) for i in range(G_size) for j in range(H_size)])
    rng = np.random.default_rng(seed=30)
    rng.shuffle(permutation)
    permutation = np.reshape(permutation, (G_size, H_size, 2))
    return permutation

HG_sizes = [(11,7), (10, 5), (6, 8), (6, 9)]

@pytest.mark.parametrize('G_size,H_size', HG_sizes)
def test_product_permutation_route(G_size, H_size):
    # Test that the routing returned by product permutation route is congestion-free

    for _ in range(100):
        permutation = _random_permutation(G_size, H_size)
        routing_row = np.reshape(product_permutation_route(permutation), (G_size, H_size, 1))
        
        # Compute the route
        route = np.concatenate([permutation, routing_row], axis=2)

        # Route along cols
        for j in range(H_size):
            col = [tuple(route[i,j,:]) for i in range(G_size)]
            col.sort(key=lambda x: x[2])
            for i in range(G_size):
                route[i,j,:] = col[i]

        # Route along rows
        for i in range(G_size):
            row = [tuple(route[i,j,:]) for j in range(H_size)]
            row.sort(key=lambda x: x[1])
            for j in range(H_size):
                route[i,j,:] = row[j]

        # Route along columns
        for j in range(H_size):
            col = [tuple(route[i,j,:]) for i in range(G_size)]
            col.sort(key=lambda x: x[0])
            for i in range(G_size):
                route[i,j,:] = col[i]

        # check
        for i in range(G_size):
            for j in range(H_size):
                assert tuple(route[i,j,:2]) == (i,j)

@pytest.mark.parametrize('G_size,H_size', HG_sizes)
def test_grid_permutation_route(G_size, H_size):

    for _ in range(100):
        permutation = _random_permutation(G_size, H_size)
        swap_schedule = grid_permutation_route(np.copy(permutation))
        assert_disjoint_swaps(swap_schedule)
        execute_swaps(permutation, swap_schedule)
        for i in range(G_size):
            for j in range(H_size):
                assert np.all(permutation[i,j,:] == [i,j])

        