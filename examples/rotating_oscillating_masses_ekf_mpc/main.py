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
import time
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc


""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules:
"""

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from examples.rotating_oscillating_masses_ekf_mpc.template_ekf import template_ekf

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
#mhe = template_mhe(model)

# setting up estimator
ekf = template_ekf(model=model)


"""
Set initial state
"""
np.random.seed(99)

# Use different initial state for the true system (simulator) and for MHE / MPC
x0_true = np.random.rand(model.n_x)-0.5
x0 = np.zeros(model.n_x)

mpc.x0 = x0
simulator.x0 = x0_true
ekf.x0 = x0
#estimator.p_est0 = 1e-4

# Set initial guess for MHE/MPC based on initial state.
mpc.set_initial_guess()
ekf.set_initial_guess()

"""
Setup graphic:
"""
"""
color = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots(5,1, sharex=True, figsize=(10, 9))

mpc_plot = do_mpc.graphics.Graphics(mpc.data)
#mhe_plot = do_mpc.graphics.Graphics(mhe.data)
sim_plot = do_mpc.graphics.Graphics(simulator.data)

ax[0].set_title('controlled position:')
mpc_plot.add_line('_x', 'phi_2', ax[0])
mpc_plot.add_line('_tvp', 'phi_2_set', ax[0], color=color[0], linestyle='--', alpha=0.5)

ax[0].legend(
    mpc_plot.result_lines['_x', 'phi_2']+mpc_plot.result_lines['_tvp', 'phi_2_set']+mpc_plot.pred_lines['_x', 'phi_2'],
    ['Recorded', 'Setpoint', 'Predicted'], title='Disc 2')

ax[1].set_title('uncontrolled position:')
mpc_plot.add_line('_x', 'phi_1', ax[1])
mpc_plot.add_line('_x', 'phi_3', ax[1])

ax[1].legend(
    mpc_plot.result_lines['_x', 'phi_1']+mpc_plot.result_lines['_x', 'phi_3'],
    ['Disc 1', 'Disc 3']
    )

ax[2].set_title('Inputs:')
mpc_plot.add_line('_u', 'phi_m_set', ax[2])

ax[3].set_title('Estimated angular velocity:')
sim_plot.add_line('_x', 'dphi', ax[3])
#mhe_plot.add_line('_x', 'dphi', ax[3])


ax[4].set_title('Estimated parameters:')
sim_plot.add_line('_p', 'Theta_1', ax[4])
#mhe_plot.add_line('_p', 'Theta_1', ax[4])

#for mhe_line_i, sim_line_i in zip(mhe_plot.result_lines.full, sim_plot.result_lines.full):
#    mhe_line_i.set_color(sim_line_i.get_color())
#    sim_line_i.set_alpha(0.5)
#    sim_line_i.set_linewidth(5)

ax[0].set_ylabel('disc \n angle [rad]')
ax[1].set_ylabel('disc \n angle [rad]')
ax[2].set_ylabel('motor \n angle [rad]')
ax[3].set_ylabel('angle \n velocity [rad/2]')
ax[4].set_ylabel('mass inertia')
ax[3].set_xlabel('time [s]')

for ax_i in ax:
    ax_i.axvline(1.0)

fig.tight_layout()
plt.ion()
"""
# ekf
q = 0 * np.ones(model.x.shape)
r = 1e-2 * np.ones(model.y.shape)
Q = np.diag(q.flatten())
R = np.diag(r.flatten())

"""
Run MPC main loop:
"""

x_data = []
x_hat_data = [x0.reshape((-1, 1))]
#%%
for k in range(50):
    u0 = mpc.make_step(x0)
    # Simulate with process and measurement noise

    y_next = simulator.make_step(u0, v0=1*np.random.randn(model.n_v,1))
    #x0 = mhe.make_step(y_next)
    x0 = ekf.make_step(y_next = y_next, u_next = u0, Q_k=Q, R_k=R)

    x_data.append(simulator.data._x[-1].reshape((-1, 1)))
    x_hat_data.append(x0.reshape((-1, 1)))

    """
    if show_animation:
        mpc_plot.plot_results()
        mpc_plot.plot_predictions()
        #mhe_plot.plot_results()
        sim_plot.plot_results()

        mpc_plot.reset_axes()
        #mhe_plot.reset_axes()
        sim_plot.reset_axes()
        plt.show()
        plt.pause(0.01)
    """


#input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, ekf, simulator], 'rot_oscillating_masses')


def visualize(x_data, x_hat_data):
    fig, ax = plt.subplots(model.n_x)
    fig.suptitle('EKF Observer')

    for i in range(model.n_x):
        ax[i].plot(x_data[i, :], label='real state')
        ax[i].plot(x_hat_data[i, :],"r--", label='estimated state')
        ax[i].set_xticklabels([])

    ax[-1].set_xlabel('time_steps')
    fig.legend()
    plt.show()

    input('Press any key to exit.')

x_data = np.concatenate(x_data, axis=1)
x_hat_data = np.concatenate(x_hat_data, axis=1)
visualize(x_data, x_hat_data)