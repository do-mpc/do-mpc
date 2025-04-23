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
store_results = True

# setting up the models
model,linearmodel = template_model()

# setting up lqr controller for the discrete linear model
lqr = template_lqr(linearmodel)

# setting up a simulator, given the linear model
simulator = template_simulator(model)

# Set the initial state of simulator:
C_a0 = 0
C_b0 = 0
T_R0 = 387.05
T_J0 = 387.05
x0 = np.array([C_a0, C_b0, T_R0, T_J0]).reshape(-1,1)

# pushing initial condition to simulator
simulator.x0 = x0

# Steady state values
F_ss = 0.002365    # [m^3/min]
Q_ss = 18.5583     # [kJ/min]
C_ass = 1.6329     # [kmol/m^3]
C_bss = 1.1101     # [kmolm^3]
T_Rss = 398.6581   # [K]
T_Jss = 397.3736   # [K]

# sets the desired operating point for the lqr
uss = np.array([[F_ss],[Q_ss]])
xss = np.array([[C_ass],[C_bss],[T_Rss],[T_Jss]])
lqr.set_setpoint(xss=xss,uss=uss)

# simulation of the plant
for k in range(200):
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
    do_mpc.data.save_results([simulator], 'results_CSTR_LQR')
