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

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from template_ekf import template_ekf


""" User settings: """
show_animation = False
store_results = False
show_plot =False

"""
Get configured do-mpc modules:
"""
model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
#estimator = do_mpc.estimator.StateFeedback(model)

# setting up estimator
estimator = template_ekf(model)


# setting up model variances with a generic value
q = 1 * np.ones(model.x.shape)
r = 0.01 * np.ones(model.y.shape)
Q = np.diag(q.flatten())
R = np.diag(r.flatten())


"""
Set initial state
"""
np.random.seed(99)

e = np.ones([model.n_x,1])
x0 = np.random.uniform(-3*e,3*e) # Values between +3 and +3 for all states
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

estimator.set_initial_guess()

"""
Setup graphic:
"""
if show_plot:
    fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
    plt.ion()

"""
Run MPC main loop:
"""
x_data = []
x_hat_data = [x0]
for k in range(50):
    u0 = mpc.make_step(x0)
    #y_next = simulator.make_step(u0, v0 = 0.001*np.random.randn(model.n_v,1))
    y_next = simulator.make_step(u0, v0=0.01 * np.random.randn(model.n_v, 1))
    #x0 = estimator.make_step(y_next)
    x0 = estimator.make_step(y_next = y_next, u_next = u0, Q_k = Q, R_k = R)

    x_data.append(simulator.data._x[-1].reshape((-1, 1)))
    x_hat_data.append(x0)

    if show_animation and show_plot:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

#input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'oscillating_masses')


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
