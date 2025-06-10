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

# imports
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import sys
import os
rel_do_mpc_path = os.path.join('..', '..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer
import matplotlib.pyplot as plt

# local imports
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

# user settings
show_animation = False
store_results = False

# setting up the model
model = template_model()

# setting up a mpc controller, given the model
mpc = template_mpc(model)

# setting up a simulator, given the model
simulator = template_simulator(model)

# setting up an estimator, given the model
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
X_p_0 = 0  # Initial x position [m]
Y_P_0 = 0  # Initial y position [m]
Psi_0 = 0  # Intial yaw angle [rad]
V  = 0.1  # Initial velocity x-axis [m/s]
x0 = np.array([X_p_0, Y_P_0, Psi_0, V]).reshape(-1, 1)

# pushing initial condition to mpc and the simulator
mpc.x0 = x0
simulator.x0 = x0

# setting up initial guesses
mpc.set_initial_guess()

# simulation of the plant
timer = Timer()
optimal_control = []
optimal_states = []
optimal_states.append(x0)
for k in range(200):

    # for the current state x0, mpc computes the optimal control action u0
    timer.tic()
    u0 = mpc.make_step(x0)
    timer.toc()

    # for the current state u0, computes the next state y_next
    y_next = simulator.make_step(u0)

    # for the current state y_next, estimates the next state x0
    x0 = estimator.make_step(y_next)

    # storage
    optimal_control.append(u0)
    optimal_states.append(x0)

# make plots
optimal_control = np.array(optimal_control)
plt.plot(optimal_control[:, 0], label='Delta')
plt.plot(optimal_control[:, 1], label='Acc')
plt.legend()
plt.show()

optimal_states = np.array(optimal_states)
plt.plot(optimal_states[:, 0], label='X_p')
plt.plot(optimal_states[:, 1], label='Y_p')
plt.plot(optimal_states[:, 2], label='Psi')
plt.plot(optimal_states[:, 3], label='V')

plt.legend()
plt.show()

plt.plot(optimal_states[:, 0], optimal_states[:, 1])
plt.show()
