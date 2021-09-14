import sys
sys.path.append('../../')
import do_mpc

import numpy as np
import pdb


sp = do_mpc.sampling.SamplingPlanner()


sp.set_sampling_var('alpha', np.random.randn)
sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

plan = sp.gen_sampling_plan('test', n_samples=10)


sampler = do_mpc.sampling.Sampler(plan)
