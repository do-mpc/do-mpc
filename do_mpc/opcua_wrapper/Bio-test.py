
#%%
import numpy as np
import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
sys.path.append('../../../')

# Import do_mpc package:
import do_mpc

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# States struct (optimization variables):
X_s = model.set_variable('_x',  'X_s')
S_s = model.set_variable('_x',  'S_s')
P_s = model.set_variable('_x',  'P_s')
V_s = model.set_variable('_x',  'V_s')

# Input struct (optimization variables):
inp = model.set_variable('_u',  'inp')

# Certain parameters
mu_m  = 0.02
K_m   = 0.05
K_i   = 5.0
v_par = 0.004
Y_p   = 1.2

# Uncertain parameters:
Y_x  = model.set_variable('_p',  'Y_x')
S_in = model.set_variable('_p', 'S_in')

# Auxiliary term
mu_S = mu_m*S_s/(K_m+S_s+(S_s**2/K_i))

# Differential equations
model.set_rhs('X_s', mu_S*X_s - inp/V_s*X_s)
model.set_rhs('S_s', -mu_S*X_s/Y_x - v_par*X_s/Y_p + inp/V_s*(S_in-S_s))
model.set_rhs('P_s', v_par*X_s - inp/V_s*P_s)
model.set_rhs('V_s', inp)

# Build the model
model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 20,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 1.0,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

mterm = -model.x['P_s'] # terminal cost
lterm = -model.x['P_s'] # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(inp=1.0) # penalty on input changes

# lower bounds of the states
mpc.bounds['lower', '_x', 'X_s'] = 0.0
mpc.bounds['lower', '_x', 'S_s'] = -0.01
mpc.bounds['lower', '_x', 'P_s'] = 0.0
mpc.bounds['lower', '_x', 'V_s'] = 0.0

# upper bounds of the states
mpc.bounds['upper', '_x','X_s'] = 3.7
mpc.bounds['upper', '_x','P_s'] = 3.0

# upper and lower bounds of the control input
mpc.bounds['lower','_u','inp'] = 0.0
mpc.bounds['upper','_u','inp'] = 0.2

Y_x_values = np.array([0.5, 0.4, 0.3])
S_in_values = np.array([200.0, 220.0, 180.0])

mpc.set_uncertainty_values(Y_x = Y_x_values, S_in = S_in_values)

mpc.setup()
#%%