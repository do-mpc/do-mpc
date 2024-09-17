"""
Date: 2023-10-19
Author: Lukas Lüken

Script to generate the sampling plans for the nonlinear double integrator nmpc.

"""
# %% Imports
import do_mpc
import numpy as np

# %% Config
#####################################################

# Samples
n_samples = 2000
data_dir = './sampling/'
sampling_plan_name = 'sampling_plan'+'_n'+str(n_samples)
overwrite = True
id_precision = np.ceil(np.log10(n_samples)).astype(int)

# Params
lbx = np.array([[-10.0], [-10.0]])
ubx = np.array([[10.0], [10.0]])
lbu = np.array([[-2.0]])
ubu = np.array([[2.0]])

#####################################################

# %% Functions

def gen_x0():
    x0 = np.random.uniform(lbx, ubx)
    return x0

def gen_u_prev():
    u_prev = np.random.uniform(lbu, ubu)
    return u_prev


# %% 
# Sampling Plan

assert n_samples<= 10**(id_precision+1), "Not enough ID-digits to save samples"

# Initialize sampling planner
sp = do_mpc.sampling.SamplingPlanner()
sp.set_param(overwrite=overwrite)
sp.set_param(id_precision=id_precision)
sp.data_dir = data_dir

# Set sampling vars
sp.set_sampling_var('x0', gen_x0)
sp.set_sampling_var('u_prev', gen_u_prev)

# Generate sampling plan
plan = sp.gen_sampling_plan(n_samples=n_samples)

# Export
sp.export(sampling_plan_name)

# import pickle as pkl
# with open('./sampling_test/test_sampling_plan.pkl','rb') as f:
#     plan = pkl.load(f)