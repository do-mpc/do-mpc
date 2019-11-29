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
import scipy.io as sio
import matplotlib.pyplot as plt
plt.ion()

from template_model import template_model
from template_optimizer import template_optimizer
from template_simulator import template_simulator

C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

data = sio.loadmat('mpc_result.mat')
x_mpc = data['mpc_states']
u_mpc = data['mpc_control']

# Build model
model = template_model()
simulator = template_simulator(model)

x_list = []
x_list.append(x0.reshape(1,-1))

for i in range(1,u_mpc.shape[0]):
    simulator.sim_x_num['_x'] = x_list[-1].reshape(-1,1)
    simulator.sim_p_num['_u'] = u_mpc[i,:].reshape(-1,1)
    simulator.sim_p_num['_z'] = np.array([1.0]).reshape(-1,1)
    simulator.sim_p_num['_tvp'] = np.array([1.0,1.0]).reshape(-1,1)
    simulator.sim_p_num['_p'] = np.array([1.0,1.0]).reshape(-1,1)
    x_new = simulator.simulate()
    x_list.append(np.reshape(x_new,(1,-1)))

x_sim = np.vstack(x_list)

# plot states
for i in range(4):
    plt.figure()
    plt.plot(x_mpc[:,i])
    plt.plot(x_sim[:,i],':')
