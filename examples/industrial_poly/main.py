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
import pickle
import time


from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator


""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
x0 = model['x'](0)

delH_R_real = 950.0
c_pR = 5.0

x0['m_W'] = 10000.0
x0['m_A'] = 853.0
x0['m_P'] = 26.5

x0['T_R'] = 90.0 + 273.15
x0['T_S'] = 90.0 + 273.15
x0['Tout_M'] = 90.0 + 273.15
x0['T_EK'] = 35.0 + 273.15
x0['Tout_AWT'] = 35.0 + 273.15
x0['accum_monom'] = 300.0
x0['T_adiab'] = x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']



mpc.set_initial_state(x0, reset_history=True)
simulator.set_initial_state(x0, reset_history=True)

# Initialize graphic:
graphics = do_mpc.graphics.Graphics(mpc.data)


fig, ax = plt.subplots(5, sharex=True)
plt.ion()
# Configure plot:
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[0])
graphics.add_line(var_type='_x', var_name='T_adiab', axis=ax[1])
graphics.add_line(var_type='_x', var_name='accum_monom', axis=ax[2])
graphics.add_line(var_type='_u', var_name='m_dot_f', axis=ax[3])
graphics.add_line(var_type='_u', var_name='T_in_M', axis=ax[4])
graphics.add_line(var_type='_u', var_name='T_in_EK', axis=ax[4])

ax[0].set_ylabel('$T_R$ [K]')
ax[1].set_ylabel('$T_{adiab}$ [K]')
ax[2].set_ylabel('acc. monom')
ax[3].set_ylabel('m_dot_f')
ax[4].set_ylabel('$T^{in}_M, T^{in}_{EK} $ [K]')
ax[4].set_xlabel('Time [h]')

# Draw the constraints (disable if not implemented)
ax[0].plot([0,1.80],[361.15, 361.15], color='red', linestyle='-.')  # lower constraint = 88 °C
ax[0].plot([0,1.80],[365.15, 365.15], color='red', linestyle='-.')  # lower constraint = 92 °C
# The nexst line represents the soft constraint
ax[1].plot([0,1.80],[381.15,381.15], color='orange', linestyle='-.')  # sfctr = 108 °C
# And finally also control value limitations
ax[4].plot([0,1.80],[333.15, 333.15], color='red', linestyle='-.')     # lower = 60 °C
ax[4].plot([0,1.80],[373.15, 373.15], color='red', linestyle='-.')   # upper = 100 °C

fig.align_ylabels()
plt.ion()

for k in range(100):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'industrial_poly')
