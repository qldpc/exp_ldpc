import qldpc
from qldpc.misc import p_sweep_main

def constant_pheno_prior(p,_,__):
    return 2/3*p

if __name__ == '__main__':
    noise_model_args = lambda p_ph: {'p':p_ph, 'pm':p_ph}
    noise_model = qldpc.noise_model.depolarizing_noise
    pheno_prior = constant_pheno_prior
    p_sweep_main(noise_model_args, noise_model, pheno_prior, pheno_prior)
