"""
Date: 2023-10-19
Author: Lukas Lüken

Script for sampling the nonlinear double integrator nmpc based on given sampling plan.

"""

# %% Imports
import numpy as np
import do_mpc
from pathlib import Path
import pandas as pd
# import time
from timeit import default_timer as timer
import pickle as pkl

# Control Problem
from nl_double_int_nmpc.template_model import template_model
from nl_double_int_nmpc.template_mpc import template_mpc
from nl_double_int_nmpc.template_simulator import template_simulator

from nlp_handler import NLPHandler

file_pth = Path(__file__).parent.resolve()
print("Filepath: ",file_pth)

# %% Config
#####################################################

silence_solver = True
return_full_mpc_data = True

# Samples
## Where sampling files are stored (sampling plans, datasets and samples)? (DEFAULT)
data_dir = file_pth.joinpath('./sampling')
## How are samples named? (DEFAULT)
sample_name = 'sample'
suffix = '_n4000'
sampling_plan_name = 'sampling_plan'+suffix #'sampling_plan'+suffix

overwrite_sampler = False
samples_dir = data_dir.joinpath('samples'+suffix)

# Data
test_run = False
# filter_success_runs = False
data_file_name = 'data'

#####################################################
# %% MPC
# Initialize model, simulator, mpc, estimator 
model = template_model()
mpc = template_mpc(model,silence_solver=silence_solver)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)
# Assertion for scaling
for val in [mpc._x_scaling.cat, mpc._p_scaling.cat, mpc._u_scaling.cat]:
    assert (np.array(val)==1).all(), "you have to consider scaling: change opt_x_num to consider scaled values"

# %% NLP Handler
# setup NLP Handler
nlp_handler = NLPHandler(mpc)

# %% Functions

# Sampling functions
def run_mpc_one_step(x0, u_prev, return_full=False):
    mpc.reset_history()
    mpc.x0 = x0
    mpc.u0 = u_prev
    mpc.set_initial_guess()

    start = timer()
    u0 = mpc.make_step(x0)
    end = timer()

    stats = {}
    stats["t_make_step"] = end-start
    stats["success"] = mpc.solver_stats["success"]
    stats["iter_count"] = mpc.solver_stats["iter_count"]

    if "t_wall_total" in mpc.solver_stats:
        stats["t_wall_total"] = mpc.solver_stats["t_wall_total"]
    else:
        stats["t_wall_total"] = np.nan

    if return_full:
        ### get solution
        nlp_sol, p_num = nlp_handler.get_mpc_sol(mpc)
        z_num = nlp_handler.extract_numeric_primal_dual_sol(nlp_sol)
        ### reduced solution
        # z_num, p_num = nlp_handler.get_reduced_primal_dual_sol(nlp_sol,p_num)
        return u0, stats, np.array(z_num), np.array(p_num), mpc.data
    else:
        return u0, stats

# Sampling function
def sample_function(x0, u_prev):
    return run_mpc_one_step(x0, u_prev, return_full=return_full_mpc_data)

# %% Sampling Plan
# Import sampling plan
# with open(data_dir+sampling_plan_name+'.pkl','rb') as f:
with open(data_dir.joinpath(sampling_plan_name+'.pkl'),'rb') as f:
    plan = pkl.load(f)

# %% Sampler
sampler = do_mpc.sampling.Sampler(plan)
sampler.data_dir = str(samples_dir)+'/'
sampler.set_param(overwrite=overwrite_sampler)
sampler.set_param(sample_name=sample_name)

sampler.set_sample_function(sample_function)

# %% Main - Sample Data
if test_run:
    sampler.sample_idx(0)
else:
    sampler.sample_data()

# %% Data Handling
dh = do_mpc.sampling.DataHandler(plan)

dh.data_dir = str(samples_dir)+'/'
dh.set_param(sample_name = sample_name)
dh.set_post_processing('u0', lambda x: x[0].item())
dh.set_post_processing('status', lambda x: x[1]["success"])
dh.set_post_processing('t_make_step', lambda x: x[1]["t_make_step"])
dh.set_post_processing('t_wall', lambda x: x[1]["t_wall_total"])
dh.set_post_processing('iter_count', lambda x: x[1]["iter_count"])
if return_full_mpc_data == True:
    dh.set_post_processing('z_num', lambda x: x[2])
    dh.set_post_processing('p_num', lambda x: x[3])
    dh.set_post_processing('mpc_data', lambda x: x[4])

# if filter_success_runs:
#     df = pd.DataFrame(dh.filter(output_filter = lambda status: status==True))
# else:
#     df = pd.DataFrame(dh[:])

# n_data = df.shape[0]

# # %% Save
# if filter_success_runs:
#     df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')
# else:
#     df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')

# %% Save
# Filter opt and Save
df = pd.DataFrame(dh.filter(output_filter = lambda status: status==True))
n_data = df.shape[0]
df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_opt' + '.pkl')

# Save all
df = pd.DataFrame(dh[:])
n_data = df.shape[0]
df.to_pickle(str(data_dir) +'/' + data_file_name + '_n{}'.format(n_data) + '_all' + '.pkl')
