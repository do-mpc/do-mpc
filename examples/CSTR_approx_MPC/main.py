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
from pathlib import Path
import matplotlib.pyplot as plt
import pickle
import time
import matplotlib
#import torch

#from do_mpc.approximateMPC.approx_MPC import ApproxMPC, ApproxMPCSettings
from do_mpc.approximateMPC.sampling import Sampler
#from do_mpc.approximateMPC.approx_MPC import ApproxMPC, Trainer,FeedforwardNN
#from do_mpc.approximateMPC import approx_mpc_training
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" User settings: """
show_animation = True
store_results = False
matplotlib.use('TkAgg')

model = template_model()
mpc = template_mpc(model,silence_solver=True)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)
u0=np.array([[0],[0]])
mpc.x0 = x0
simulator.x0 = x0

mpc.set_initial_guess()
lbx = np.array([[0.1], [0.1], [50], [50]])
ubx = np.array([[2], [2], [140], [140]])
lbu = np.array([[5], [-8500]])
ubu = np.array([[100], [0]])
n_samples=10
#file_pth = Path(__file__).parent.resolve()
data_dir = './sampling'
#data_dir = file_pth.joinpath(data_dir)
#net=FeedforwardNN(n_in=mpc.model.n_x+mpc.model.n_u,n_out=mpc.model.n_u)
#approx_mpc = ApproxMPC(net)
sampler=Sampler()
#trainer=Trainer(approx_mpc)
sampler.default_sampling(mpc,n_samples,lbx,ubx,lbu,ubu)
#x=np.stack((x0,u0))
#y=approx_mpc.make_step(x)


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

timer = Timer()

for k in range(50):
    timer.tic()
    u0 = mpc.make_step(x0)
    timer.toc()
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

timer.info()
timer.hist()

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'CSTR_robust_MPC')
