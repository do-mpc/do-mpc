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

import os
import sys
import torch
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

rel_do_mpc_path = os.path.join('..','..','..') 
sys.path.append(rel_do_mpc_path)
import do_mpc

from template_converter import template_converter
from template_model import template_model
from template_simulator import template_simulator
from template_mpc import template_mpc

# Initializing the neural network model
layers = []
layers.append(torch.nn.Linear(3, 10))

# activation layer
layers.append(torch.nn.Tanh())

# Output layer (from the last hidden layer to output_size)
layers.append(torch.nn.Linear(10, 2))

# Combine all layers into a sequential model
nn_model = torch.nn.Sequential(*layers)

# load th pretrained model weights
nn_model.load_state_dict(torch.load('model_weights.pth'))

# initialising the conversion
# to do-mpc model
surrogate_model = template_converter(nn_model)

surrogate_mpc = template_mpc(model=surrogate_model)

# setting up surrogate simulator
surrogate_simulator = do_mpc.simulator.Simulator(model=surrogate_model)
surrogate_simulator.set_param(t_step=0.1)
surrogate_simulator.setup()

# setting up real model
real_model = template_model()

# setting up real simulator
real_simulator = template_simulator(model=real_model)

# constraints
lbx = np.array([-0.01, -2.65/100])
ubx = np.array([0.01, 2.65/100])

# random state inside the state boundaries
x0 = np.array([0.004, 0.0])
real_simulator.x0 = x0
surrogate_simulator.x0 = x0
surrogate_mpc.x0 = x0

real_simulator.set_initial_guess()
surrogate_simulator.set_initial_guess()
surrogate_mpc.set_initial_guess()

lbu = np.array([-0.1])
ubu = np.array([0.1])

# data storage
x0_real_data = [x0.reshape((1, -1))]
x0_surrogate_data = [x0.reshape((1, -1))]
u0_data = []

# main simulation loop
for _ in range(50):

    # optimal input form mpc
    u_0 = surrogate_mpc.make_step(x0=x0)

    # simulation steps
    x0_real = real_simulator.make_step(u0=u_0)
    x0_surrogate = surrogate_simulator.make_step(u0=u_0)

    # data stored for plots
    x0_real_data.append(x0_real.reshape((1, -1)))
    x0_surrogate_data.append(x0_surrogate.reshape((1, -1)))
    u0_data.append(u_0.reshape((1, -1)))

# plots
x0_real_data = np.vstack(x0_real_data)
x0_surrogate_data = np.vstack(x0_surrogate_data)
u0_data = np.vstack(u0_data)

fig, ax = plt.subplots(3, figsize=(10, 6))

ax[0].plot(real_simulator.data['_time'], x0_real_data[:-1, 0], 'b', label='Real Model')
ax[0].plot(surrogate_simulator.data['_time'], x0_surrogate_data[:-1, 0], 'r', label='Surrogate Model')
ax[0].set_ylabel('Position [m]')
ax[0].set_title('States:')

ax[1].plot(real_simulator.data['_time'], x0_real_data[:-1, 1], 'b')
ax[1].plot(surrogate_simulator.data['_time'], x0_surrogate_data[:-1, 1], 'r')
ax[1].set_ylabel('Velocity [m/s]')


ax[2].plot(surrogate_simulator.data['_time'], u0_data[:, 0],'g--', label='Random Input')
ax[2].set_ylabel('Force [N]')
ax[2].set_xlabel('Time [s]')
ax[2].set_title('Inputs:')

fig.legend()
fig.suptitle('Real vs Surrogate Model', fontsize=16)
plt.tight_layout()
plt.show()