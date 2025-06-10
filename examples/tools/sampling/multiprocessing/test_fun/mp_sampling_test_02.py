import sys
sys.path.append('../../../../../')
import do_mpc
import time
import os

import numpy as np
import pdb
import multiprocessing as mp
from do_mpc.tools import load_pickle


def info():
    print(mp.current_process())
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def sample_function(alpha, beta):
    #info()
    time.sleep(1)
    return alpha*beta

def main():
    plan = load_pickle('./samples/sp_mp_test.pkl')

    sampler = do_mpc.sampling.Sampler(plan)
    sampler.set_param(overwrite = True)
    sampler.set_param(print_progress = True)
    sampler.data_dir = './samples/'

    sampler.set_sample_function(sample_function)

    tic = time.time()

    with mp.Pool(processes=2) as pool:
        p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))

    toc = time.time()

    print('Elapsed time: {}'.format(toc-tic))

if __name__ == '__main__':
    main()
