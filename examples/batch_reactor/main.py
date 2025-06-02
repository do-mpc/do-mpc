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
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import matplotlib.pyplot as plt

# local imports
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

# user settings
show_animation = True
store_results = False

# setting up the model
model = template_model()

# setting up a mpc controller, given the model
mpc = template_mpc(model)

# setting up a simulator, given the model
simulator = template_simulator(model)

# setting up an estimator, given the model
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc, simulator and estimator:
X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
S_s_0 = 0.5 # This is the controlled variable [mol/l]
P_s_0 = 0.0 #[C]
V_s_0 = 120.0 #[C]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

# pushing initial condition to mpc, simulator and estimator
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# setting up initial guesses
mpc.set_initial_guess()

# Initialize graphic:
fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(8,5))
plt.ion()

# simulation of the plant
for k in range(150):

    # for the current state x0, mpc computes the optimal control action u0
    u0 = mpc.make_step(x0)

    # for the current state u0, computes the next state y_next
    y_next = simulator.make_step(u0)

    # for the current state y_next, estimates the next state x0
    x0 = estimator.make_step(y_next)

    # update the graphics
    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'batch_reactor_MPC')
