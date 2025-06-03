import sys
sys.path.append('../../../../../')
import do_mpc

import time
import os

import numpy as np
import pdb
import multiprocessing as mp
from functools import partial
from do_mpc.tools import load_pickle


sys.path.append('../../../oscillating_masses_discrete/')
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" Load necessary configuration """
model = template_model()
mpc = template_mpc(model)
estimator = do_mpc.estimator.StateFeedback(model)
simulator = template_simulator(model)

def info():
    print(mp.current_process())
    print('parent process:', os.getppid())
    print('process id:', os.getpid())


def sample_function(X0):
    info()

    mpc.reset_history()
    simulator.reset_history()
    estimator.reset_history()

    # set initial values and guess
    x0 = X0
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    mpc.set_initial_guess()

    # run the closed loop for 150 steps
    for k in range(100):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
    return simulator.data



def main():
    plan = load_pickle('./samples/closed_loop_mp.pkl')

    sampler = do_mpc.sampling.Sampler(plan)
    sampler.set_param(overwrite = True)
    sampler.set_param(print_progress = False)
    sampler.data_dir = './samples/'


    sampler.set_sample_function(sample_function)

    tic = time.time()

    with mp.Pool(processes=4) as pool:
        p = pool.map(sampler.sample_idx, list(range(sampler.n_samples)))

    toc = time.time()

    print('Elapsed time: {}'.format(toc-tic))

    # DataHandling
    dh = do_mpc.sampling.DataHandler(plan)

    dh.data_dir = './samples/'
    dh.set_param(overwrite = True)

    res0 = dh[0]
    res1 = dh[:]
    res2 = dh.filter(lambda X0: X0[0]<1.2)

    dh.set_post_processing('input', lambda data: data['_u', 'u'])
    dh.set_post_processing('state', lambda data: data['_x', 'x'])

    res3 = dh[0]
    res4 = dh[:]
    res5 = dh.filter(lambda X0: X0[0]<1.2)
    res6 = dh.filter(lambda id: id=='001')

    # Return nothing:
    res7 = dh.filter(lambda X0: X0[0]>10)
    res8 = dh[8]


if __name__ == '__main__':
    main()
