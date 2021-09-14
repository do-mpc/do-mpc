import sys
sys.path.append('../../')
sys.path.append('../batch_reactor/')
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

    X_s_0 = np.random.uniform(0.9,1.1)
    S_s_0 = np.random.uniform(0.4,0.6)
    P_s_0 = 0.0
    V_s_0 = np.random.uniform(115.0,125.0)

    return np.reshape([X_s_0, S_s_0, P_s_0, V_s_0], (-1, 1))

def gen_uncertainty_values():

    Y_x = np.random.uniform(0.3,0.5)
    S_in = np.random.uniform(180.0,220.0)

    return np.reshape([Y_x,S_in],(-1,1))

# Add variables
sp.set_sampling_var('x0', gen_initial_states)
sp.set_sampling_var('p', gen_uncertainty_values)

plan = sp.gen_sampling_plan('batch_reactor', n_samples = 10)


""" Execute sampling plan """
# Define the sampling function, in this case closed-loop mpc runs
""" Load necessary configuration """
model = template_model()
mpc = template_mpc(model)
estimator = do_mpc.estimator.StateFeedback(model)

def run_closed_loop(x0,p):

    # initialize simulator with uncertainties from sampling plan
    simulator = template_simulator(model, p)

    # set initial values and guess
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

# set sampling function
sampler.set_sample_function(run_closed_loop)

# Generate the samples
sampler.sample_data()
