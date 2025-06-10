import sys
sys.path.append('../../../../../')
import do_mpc
import time
import os

import numpy as np
import pdb

def main():
    np.random.seed(123)

    sp = do_mpc.sampling.SamplingPlanner()
    sp.data_dir = './samples/'

    # Generate sampling function for initial states
    def gen_initial_states():

        x0 = np.random.uniform(-3*np.ones((4,1)),3*np.ones((4,1)))

        return x0


    # Add variables
    sp.set_sampling_var('X0', gen_initial_states)

    sp.set_param(overwrite=True)

    plan = sp.gen_sampling_plan(n_samples = 10)

    sp.export('closed_loop_mp')



if __name__ == '__main__':
    main()
