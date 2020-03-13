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


model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
x_0 = -1.0 # This is the initial concentration inside the tank [mol/l]
v_0 = 0.0 # This is the controlled variable [mol/l]
theta_0 = 0.3 #[C]
omega_0 = 0.0 #[C]
x0 = np.array([x_0, v_0, theta_0, omega_0]).reshape(-1,1)

mpc.set_initial_state(x0, reset_history=True)
simulator.set_initial_state(x0, reset_history=True)

# Initialize graphic:
graphics = do_mpc.graphics.Graphics()


fig, ax = plt.subplots(3, sharex=True)
# Configure plot:
graphics.add_line(var_type='_x', var_name='x', axis=ax[0])
graphics.add_line(var_type='_x', var_name='theta', axis=ax[1])
graphics.add_line(var_type='_u', var_name='F', axis=ax[2])
ax[0].set_ylabel('x [m]')
ax[1].set_ylabel('$\Theta$ [°]')
ax[2].set_ylabel('$F_u$ [N]')


fig.align_ylabels()
plt.ion()

time_list = []
for k in range(int(5/mpc.t_step)):
    tic = time.time()
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    toc = time.time()
    time_list.append(toc-tic)

    # if True:
    #     graphics.reset_axes()
    #     graphics.plot_results(mpc.data, linewidth=3)
    #     graphics.plot_predictions(mpc.data, linestyle='--', linewidth=1)
    #     plt.show()
    #     input('next step')

time_arr = np.array(time_list)
print('Total run-time: {tot:5.2f} s, step-time {mean:.3f}+-{std:.3f} s.'.format(tot=np.sum(time_arr), mean=np.mean(time_arr), std=np.sqrt(np.var(time_arr))))

opti_lines = graphics.plot_results(mpc.data)
simu_lines = graphics.plot_results(simulator.data)

# plt.sca(ax[0])
# ax[0].add_artist(plt.legend(opti_lines[:0], ['x'], title='mpc', loc=1))
# plt.sca(ax[0])
# ax[0].add_artist(plt.legend(simu_lines[:0], ['x'], title='Simulator', loc=2))
# plt.show()
# input('Press any key to exit.')

# Store results for animated plotting and more
do_mpc.data.save_results([mpc, simulator], 'cartpole_results')
