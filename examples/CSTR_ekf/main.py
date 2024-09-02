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
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer

import matplotlib.pyplot as plt
import pickle
import time


from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from template_ekf import template_ekf

""" User settings: """
show_animation = False
store_results = False
plot_flag = False       # for mpc only


model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
#estimator = do_mpc.estimator.StateFeedback(model)
ekf = template_ekf(model)

# setting up model variances with a generic value
#tmp = 1
#q = 1e-3 * np.ones(model.n_x)
q = 0 * np.ones(model.n_x)
#r = 1e-2 * np.ones(model.n_y)
r = 1000 * np.ones(model.n_y)

Q = np.diag(q.flatten())
R = np.diag(r.flatten())

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

mpc.x0 = x0
simulator.x0 = x0
ekf.x0 = x0
#ekf.P0 = np.ones((1,1))

mpc.set_initial_guess()
ekf.set_initial_guess()

if plot_flag:
    # Initialize graphic:
    graphics = do_mpc.graphics.Graphics(mpc.data)


    fig, ax = plt.subplots(5, sharex=True)
    # Configure plot:
    graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
    graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
    graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
    graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
    graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
    graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
    graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
    ax[0].set_ylabel('c [mol/l]')
    ax[1].set_ylabel('T [K]')
    ax[2].set_ylabel('$\Delta$ T [K]')
    ax[3].set_ylabel('Q [kW]')
    ax[4].set_ylabel('Flow [l/h]')
    ax[4].set_xlabel('time [h]')
    # Update properties for all prediction lines:
    for line_i in graphics.pred_lines.full:
        line_i.set_linewidth(1)

    label_lines = graphics.result_lines['_x', 'C_a']+graphics.result_lines['_x', 'C_b']
    ax[0].legend(label_lines, ['C_a', 'C_b'])
    label_lines = graphics.result_lines['_x', 'T_R']+graphics.result_lines['_x', 'T_K']
    ax[1].legend(label_lines, ['T_R', 'T_K'])

    fig.align_ylabels()
    fig.tight_layout()
    plt.ion()

x_data = []
x_hat_data = [x0]

timer = Timer()

for k in range(50):
    timer.tic()
    u0 = mpc.make_step(x0)
    timer.toc()
    #y_next = simulator.make_step(u0, v0=0.001*np.random.randn(model.n_v,1))
    y_next = simulator.make_step(u0, v0=10*np.random.randn(model.n_v,1))
    #x0 = estimator.make_step(y_next)
    x0 = ekf.make_step(y_next = y_next, u_next = u0, Q_k=Q, R_k=R)

    x_data.append(simulator.data._x[-1].reshape((-1,1)))
    x_hat_data.append(x0)

    if show_animation and plot_flag:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

timer.info()
#timer.hist()



# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'CSTR_robust_MPC')

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

    #input('Press any key to exit.')

x_data = np.concatenate(x_data, axis=1)
x_hat_data = np.concatenate(x_hat_data, axis=1)
visualize(x_data, x_hat_data)

