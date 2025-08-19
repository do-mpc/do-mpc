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
import sys
import os
rel_do_mpc_path = os.path.join('..','..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import matplotlib.pyplot as plt

# local imports
from template_model import template_model
from template_lqr import template_lqr
from template_simulator import template_simulator

# user settings
store_results = False

# setting up the models
t_sample = 0.5
model, daemodel, linearmodel = template_model()
model_dc = linearmodel.discretize(t_sample)

# setting up lqr controller for the discrete linear model
lqr = template_lqr(model_dc)

# setting up a simulator, given the linear model
simulator = template_simulator(linearmodel)

# Set the initial state of mpc and simulator:
Ca0 = 1
Cb0 = 0
Ad0 = 0
Cain0 = 0
Cc0 = 0
x0 = np.array([[Ca0],[Cb0],[Ad0],[Cain0],[Cc0]])

# pushing initial condition to simulator
simulator.x0 = x0

# Set setpoints
Ca_ss = 0
Cb_ss = 2
Ad_ss = 3
Cain_ss = 0
Cc_ss = 2

# sets the desired operating point for the lqr
xss = np.array([[Ca_ss],[Cb_ss],[Ad_ss],[Cain_ss],[Cc_ss]])
uss = model_dc.get_steady_state(xss = xss)
lqr.set_setpoint(xss = xss, uss = uss)

# simulation of the plant
for k in range(50):
    u0 = lqr.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = y_next
    
# Configure plot:
fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data, figsize=(16,9))
graphics.plot_results()
graphics.reset_axes()
plt.show()

# Store results:
if store_results:
    do_mpc.data.save_results([simulator], 'results_batch_reactor_LQR_DAE')