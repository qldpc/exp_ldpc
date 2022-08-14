from cmath import log
from qldpc import *
from qldpc.noise_model import depolarizing_noise
import stim

if __name__ == '__main__':
    (checks, logicals) = biregular_hpg(12, 3, 4, seed=670235982)
    p = 0.01
    rounds = 1
    samples = 1024

    noise_model = lambda *x: depolarizing_noise(p, p, *x)
    circuit, meas_result, data_result = build_storage_simulation(rounds, noise_model, checks, use_x_logicals = False)
    
    print(checks, logicals)

    sampler = stim.Circuit('\n'.join(circuit)).compile_sampler()
    batch = sampler.sample(samples)

    code = ErrorCorrectingCode([list(r) for r in checks[0].tolil().rows], logicals[1])

    bp_decoder = FirstMinBPplusSSF.wrapped_new(code, p)

    logical_values = []
    for i in range(samples):
        data_readout = data_result(batch[i]>0)
        syndrome = code.compute_syndrome(data_readout)
        correction = code.apply_decoder(bp_decoder, syndrome)
        data_readout = correction ^ data_readout
        logicals = code.measure_logicals(data_readout)
        logical_values.append(logical_values)
    print(logical_values)

