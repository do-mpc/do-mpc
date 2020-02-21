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
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_optimizer import template_optimizer
from template_simulator import template_simulator

"""
Get configured do mpc modules:
"""

model = template_model()
mpc = template_optimizer(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
S_s_0 = 0.5 # This is the controlled variable [mol/l]
P_s_0 = 0.0 #[C]
V_s_0 = 120.0 #[C]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0]).reshape(-1,1)
mpc.set_initial_state(x0, reset_history=True)
simulator.set_initial_state(x0, reset_history=True)
estimator.set_initial_state(x0, reset_history=True)

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(model, figsize=(8,5))
plt.ion()

"""
Run MPC main loop:
"""

time_list = []
for k in range(150):
    tic = time.time()
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)
    toc = time.time()
    time_list.append(toc-tic)

    if True:
        graphics.reset_axes()
        graphics.plot_results(mpc.data, linewidth=3)
        graphics.plot_predictions(mpc.data, mpc.opt_x_num, mpc.opt_aux_num, linestyle='--', linewidth=1)
        plt.show()
        input('next step')

time_arr = np.array(time_list)
print('Total run-time: {tot:5.2f} s, step-time {mean:.3f}+-{std:.3f} s.'.format(tot=np.sum(time_arr), mean=np.mean(time_arr), std=np.sqrt(np.var(time_arr))))

simu_lines = graphics.plot_results(simulator.data)
plt.show()
input('Press any key to exit.')
