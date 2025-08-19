import sys
sys.path.append('../../../../../')
sys.path.append('../../../../oscillating_masses_discrete/')
import do_mpc

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

import numpy as np
import pdb

""" Example on how to generate data with closed loop mpc """

""" Generate sampling plan """
# Generate sampling planner
sp = do_mpc.sampling.SamplingPlanner()

# Generate sampling function for initial states
def gen_initial_states():

    x0 = np.random.uniform(-3*np.ones((4,1)),3*np.ones((4,1)))

    return x0


# Add variables
sp.set_sampling_var('X0', gen_initial_states)
sp.set_param(overwrite=True)
sp.data_dir = './samples/'

plan = sp.gen_sampling_plan(n_samples = 10)


""" Execute sampling plan """
# Define the sampling function, in this case closed-loop mpc runs
""" Load necessary configuration """
model = template_model()
mpc = template_mpc(model)
estimator = do_mpc.estimator.StateFeedback(model)
simulator = template_simulator(model)

def run_closed_loop(X0):
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
    for k in range(10):
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

    return mpc.data

# Feed plan to sampler
sampler = do_mpc.sampling.Sampler(plan)
sampler.data_dir = './samples/'

# set sampling function
sampler.set_sample_function(run_closed_loop)

# Generate the samples
sampler.sample_data()


dh = do_mpc.sampling.DataHandler(plan)


dh.set_post_processing('input', lambda data: data['_u', 'u'])
dh.set_post_processing('state', lambda data: data['_x', 'x'])

dh.data_dir = './samples/'

res = dh[:]
