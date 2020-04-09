#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

import matplotlib.pyplot as plt
import pickle
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from template_ekf import template_ekf

""" User settings: """
output_feedback = False     # Do not use the estimator in the MPC loop (only use EKF data for comparison)
show_animation  = True      # Display live animation at runtime
store_results   = False     # Do not store pickled results (if True, creates a new result folder and stores each run under different name)

model = template_model()
sim = template_simulator(model)
mpc = template_mpc(model)
ekf = template_ekf(model)

# Set the initial state of mpc and simulator:
C_a_0   = 0.8       # This is the initial concentration inside the tank [mol/l]
C_b_0   = 0.5       # This is the controlled variable [mol/l]
T_R_0   = 134.14    # [C]
T_K_0   = 130.0     # [C]
alpha_0 = 0.9       # The reaction rate coeff of A-->B to be estimated (nominal = 1)
beta_0  = 0.95      # The reaction rate coeff of B-->C to be estimated (nominal = 1)

x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

mpc.set_initial_state(x0, reset_history=True)
sim.set_initial_state(x0, reset_history=True)

"Estimator initialization: select 1) only state estimation or 2) state and parameter estimation"
"(!): Remember to change the EKF tuning accordingly in template_ekf"
#x0_hat = np.array([0.8, 0.5, 100.0, 100.0]).reshape(-1,1)          # 1) only state estimation
x0_hat = np.array([0.8, 0.5, 100.0, 100.0, alpha_0]).reshape(-1,1)  # 2) state and parameter estimation
ekf.set_initial_state(x0_hat, reset_history=True)

# Initialize graphic:
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
ekf_graphics = do_mpc.graphics.Graphics(ekf.data)

fig, ax = plt.subplots(5, sharex=True)
# Configure plot:
# 1) Create the MPC lines
mpc_graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
mpc_graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
mpc_graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
mpc_graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
mpc_graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
# 2) Add the estimated lines to be compared to the MPC results 
#   (if output_feedback = False, the lines are only for comparison purposes)
ekf_graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0], linestyle = '-.', color='#1f77b4')
ekf_graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0], linestyle = '-.', color='#ff7f0e')

ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('T [K]')
ax[2].set_ylabel('$\Delta$ T [K]')
ax[3].set_ylabel('Q [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('Time [h]')

label_lines = mpc_graphics.result_lines['_x', 'C_a']+mpc_graphics.result_lines['_x', 'C_b'] + \
              ekf_graphics.result_lines['_x', 'C_a']+ekf_graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['True $C_a$', 'True $C_b$', 'Estimated $C_a$', 'Estimated $C_b$'])
label_lines = mpc_graphics.result_lines['_x', 'T_R']+mpc_graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['$T_R$', '$T_K$'])

fig.align_ylabels()
plt.ion()

time_list = []
for k in range(100):
    
    if k==0: xk = x0; xk_hat = x0
    "Note: This is an output-feedback NMPC scheme. If state-feedback is desired, update the setting at the top of this script"   
    tic = time.time()
    "Step 1: Call the NMPC controller and obtain the new optimal inputs `uk`"
    uk = mpc.make_step(xk)
    "Step 2: Call the simulator. A predefined output vector is returned `yk`"
    yk = sim.make_step(uk)
    "Note: The EKF needs both the inputs and the current parameter values"
    pk = sim.p_fun(sim._t0).cat.toarray()
    "Step 3: Call the EKF with previously obtained data at step `k`"
    xk_hat, pk_hat = ekf.make_step(yk, uk, pk)
    "Note: If parameter estimation was selected, pk_hat will be non-empty. Otherwise pk_hat=[]"
    if output_feedback : 
        # The full state of the plant will be fed into the MPC
        xk = xk_hat
    else:
        #The estimated state will be used
        xk = sim._x0.cat
    
    toc = time.time()
    time_list.append(toc-tic)
    "The live animations can be activated at the top of the script by setting show_animation=True"
    if show_animation:
        mpc_graphics.plot_results(t_ind=k)
        ekf_graphics.plot_results(t_ind=k)
        mpc_graphics.plot_predictions(t_ind=k)
        mpc_graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

time_arr = np.array(time_list)
print('Total run-time: {tot:5.2f} s, step-time {mean:.3f}+-{std:.3f} s.'.format(tot=np.sum(time_arr), mean=np.mean(time_arr), std=np.sqrt(np.var(time_arr))))

est_lines = ekf_graphics.plot_results()
sim_lines = mpc_graphics.plot_results()

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator, estimator], 'cstr_NMPC_EKF')
