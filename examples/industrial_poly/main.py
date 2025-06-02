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

# Set the initial state of the controller and simulator:
delH_R_real = 950.0
c_pR = 5.0

# x0 is a property of the simulator - we obtain it and set values.
x0 = simulator.x0

# Set the initial state
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

# Finally, the controller gets the same initial state.
mpc.x0 = x0

# Which is used to set the initial guess:
mpc.set_initial_guess()

# Initialize graphic:
graphics = do_mpc.graphics.Graphics(mpc.data)


fig, ax = plt.subplots(5, sharex=True, figsize=(16,9))
plt.ion()
# Configure plot:
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[0])
graphics.add_line(var_type='_x', var_name='accum_monom', axis=ax[1])
graphics.add_line(var_type='_u', var_name='m_dot_f', axis=ax[2])
graphics.add_line(var_type='_u', var_name='T_in_M', axis=ax[3])
graphics.add_line(var_type='_u', var_name='T_in_EK', axis=ax[4])

ax[0].set_ylabel('T_R [K]')
ax[1].set_ylabel('acc. monom')
ax[2].set_ylabel('m_dot_f')
ax[3].set_ylabel('T_in_M [K]')
ax[4].set_ylabel('T_in_EK [K]')
ax[4].set_xlabel('time')

fig.align_ylabels()
plt.ion()

# simulation of the plant
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
