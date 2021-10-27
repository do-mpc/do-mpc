import sys
sys.path.append('../../../../')
import do_mpc

import numpy as np
import pdb

np.random.seed(123)

sp = do_mpc.sampling.SamplingPlanner()
sp.set_param(overwrite = True)


sp.set_sampling_var('alpha', np.random.randn)
sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

sp.set_param(save_format='pickle')

_ = sp.gen_sampling_plan(n_samples=10)
# Add custom cases:
_ = sp.add_sampling_case(alpha=10)
_ = sp.add_sampling_case(beta=10)
plan = sp.add_sampling_case(alpha=2, beta=2)

sampler = do_mpc.sampling.Sampler(plan)
sampler.set_param(overwrite=True)

def sample_function(alpha, beta):
    return alpha*beta

sampler.set_sample_function(sample_function)

sampler.sample_data()

dh = do_mpc.sampling.DataHandler(plan)
dh.set_param(data_dir='./sample_results/')

dh.set_post_processing('res_1', lambda x: x)
dh.set_post_processing('res_2', lambda x: x**2)


res = dh[:]
