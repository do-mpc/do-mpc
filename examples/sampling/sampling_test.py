import sys
sys.path.append('../../')
import do_mpc

import numpy as np
import pdb

np.random.seed(123)

sp = do_mpc.sampling.SamplingPlanner()
sp.set_param(overwrite = True)


sp.set_sampling_var('alpha', np.random.randn)
sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

sp.set_param(save_format='pickle')

plan = sp.gen_sampling_plan('test', n_samples=10)


sampler = do_mpc.sampling.Sampler(plan)


def sample_function(alpha, beta):
    return alpha*beta

sampler.set_sample_function(sample_function)

sampler.sample_data()




dh = do_mpc.sampling.DataHandler(plan)

dh.set_post_processing('res_1', lambda x: x)
dh.set_post_processing('res_2', lambda x: x**2)


res = dh[:]

pdb.set_trace()
