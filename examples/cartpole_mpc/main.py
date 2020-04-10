#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2020 Sergio Lucia, Alexandru Tatulea-Codrean
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

""" User settings: """
show_animation  = True      # Display live animation at runtime
store_results   = True     # Do not store pickled results (if True, creates a new result folder and stores each run under different name)

model = template_model()
sim = template_simulator(model)
mpc = template_mpc(model)
sfb = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
x_0     = -2.0 # This is the initial position of the cart on the X-axis
v_0     = 0.0  # This is the initial velocity
theta_0 = 0.0  # This is the initial vertical orientation of the pendulum
omega_0 = 0.0  # And finally the initial angular velocity
x0 = np.array([x_0, v_0, theta_0, omega_0]).reshape(-1,1)

mpc.set_initial_state(x0, reset_history=True)
sim.set_initial_state(x0, reset_history=True)

# Initialize graphic:
graphics = do_mpc.graphics.Graphics(mpc.data)


fig, ax = plt.subplots(3, sharex=True)
# Configure plot:
graphics.add_line(var_type='_x', var_name='x', axis=ax[0])
graphics.add_line(var_type='_x', var_name='theta', axis=ax[1])
graphics.add_line(var_type='_u', var_name='F', axis=ax[2])
ax[0].set_ylabel('$x_{cart}$ [m]')
ax[1].set_ylabel('$\Theta$ [°]')
ax[2].set_ylabel('$F_u$ [N]')

# Draw the references for comparison's sake
ax[0].plot([0,7],[0,0], color='#ff7f0e', linewidth = 1, linestyle='--')
ax[1].plot([0,7],[0,0], color='#ff7f0e', linewidth = 1, linestyle='--')

# Create simple legend entries
label_lines = graphics.result_lines['_x', 'x']+[ax[0].get_lines()[2]]
ax[0].legend(label_lines, ['Cart position','Reference'])
label_lines = graphics.result_lines['_x', 'theta']+ [ax[1].get_lines()[2]]
ax[1].legend(label_lines, ['Vertical angle', 'Reference'])
label_lines = graphics.result_lines['_u', 'F']
ax[2].legend(label_lines, ['Push Force'])



fig.align_ylabels()
plt.ion()

time_list = []
# Run for a predefined number of seconds, here 7 sec is enough to see the cart stabilizing
for k in range(int(7/mpc.t_step)):
    tic = time.time()
    u0 = mpc.make_step(x0)
    y_next = sim.make_step(u0)
    x0 = sfb.make_step(y_next)
    toc = time.time()
    time_list.append(toc-tic)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.001)

time_arr = np.array(time_list)
print('Total run-time: {tot:5.2f} s, step-time {mean:.3f}+-{std:.3f} s.'.format(tot=np.sum(time_arr), mean=np.mean(time_arr), std=np.sqrt(np.var(time_arr))))

graphics.plot_results(t_ind=k)

# Store results for animated plotting and more
if store_results:
    do_mpc.data.save_results([mpc, sim], 'cartpole_results')
