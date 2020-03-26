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


model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = template_ekf(model)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

mpc.set_initial_state(x0, reset_history=True)
simulator.set_initial_state(x0, reset_history=True)

x0_hat = np.array([0.5, 0.4, 100.0, 100.0]).reshape(-1,1)
estimator.set_initial_state(x0_hat, reset_history=True)

# Initialize graphic:
graphics = do_mpc.graphics.Graphics()


fig, ax = plt.subplots(4, sharex=True)
# Configure plot:
graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[2])
graphics.add_line(var_type='_u', var_name='F', axis=ax[3])
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('Temperature [K]')
ax[2].set_ylabel('Q_heat [kW]')
ax[3].set_ylabel('Flow [l/h]')

fig.align_ylabels()
plt.ion()

time_list = []
for k in range(100):
    if k==0: xk = x0; xk_hat = x0
    tic = time.time()
    uk = mpc.make_step(xk_hat)
    yk = simulator.make_step(uk)
    pk = simulator.p_fun(simulator._t0).cat.toarray()
    xk_hat = estimator.make_step(yk, uk, pk)
    toc = time.time()
    time_list.append(toc-tic)

time_arr = np.array(time_list)
print('Total run-time: {tot:5.2f} s, step-time {mean:.3f}+-{std:.3f} s.'.format(tot=np.sum(time_arr), mean=np.mean(time_arr), std=np.sqrt(np.var(time_arr))))

sim_lines = graphics.plot_results(simulator.data)
est_lines = graphics.plot_results(estimator.data)

plt.sca(ax[0])
ax[0].add_artist(plt.legend(sim_lines[:2], ['Ca', 'Cb'], title='Estimator', loc=1))
plt.sca(ax[0])
ax[0].add_artist(plt.legend(est_lines[:2], ['Ca', 'Cb'], title='Simulator', loc=2))
plt.show()
input('Press any key to exit.')

# Store results:
#do_mpc.data.save_results([mpc, simulator, estimator], 'cstr_MPC_EKF')
