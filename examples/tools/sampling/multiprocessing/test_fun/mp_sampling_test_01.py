import sys
sys.path.append('../../../../../')
import do_mpc
import time
import os

import numpy as np
import pdb
import multiprocessing as mp


def main():
    np.random.seed(123)

    sp = do_mpc.sampling.SamplingPlanner()
    sp.set_param(overwrite = True)
    sp.data_dir = './samples/'


    sp.set_sampling_var('alpha', np.random.randn)
    sp.set_sampling_var('beta', lambda: np.random.randint(0,5))

    sp.set_param(overwrite=True)

    _ = sp.gen_sampling_plan(n_samples=10)
    # Add custom cases:
    _ = sp.add_sampling_case(alpha=10)
    _ = sp.add_sampling_case(beta=10)
    _ = sp.add_sampling_case(alpha=2, beta=2)

    sp.export('sp_mp_test')



if __name__ == '__main__':
    main()
