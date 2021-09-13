import sys
sys.path.append('../../')
import do_mpc

import numpy as np


sp = do_mpc.sampling.SamplingPlanner()


#sp.set_sampling_var('alpha', np.random.randn)
#sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

#sp.gen_sampling_plan('test', n_samples=10)

plan = {'name': 'test',
    'plan':[
    {'x0': 2, 'N':10, 'noise':True },
    {'x0': 1, 'N': 5, 'noise':False}
    ]
}


sampler = do_mpc.sampling.Sampler(plan)
