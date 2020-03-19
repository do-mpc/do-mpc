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
from template_estimator import template_estimator

from opcmodules import Server, Client
from opcmodules import RealtimeSimulator, RealtimeController, RealtimeFeedback
from opcmodules import RealtimeTrigger
from template_opcua import template_opcua


model = template_model()

opc_server, opc_opts = template_opcua(model)
    
rt_simulator = template_simulator(model,opc_opts)

rt_estimator = template_estimator(model, opc_opts)

rt_controller = template_mpc(model,opc_opts)

"""
Initialization and preparation of the server data base
"""
# Set the initial state of mpc and simulator:
x0 = model._x(0)

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


# Step 1: initilize the simulator part
rt_simulator.set_initial_state(x0, reset_history=True)
rt_simulator.init_server(x0.cat.toarray().tolist())
# Step 2: initialize the stimator part (if present)
rt_estimator.init_server(x0.cat.toarray().tolist())
rt_estimator.set_initial_state(x0, reset_history=True)
# Step 3: only now can the optimizer be initialized, and a first optimization can be executed
rt_controller.set_initial_state(x0, reset_history=True)
rt_controller.init_server(rt_controller._u0.cat.toarray().tolist())

# Step 4: the cyclical operation can be safely started now
"""
Define triggers for each of the modules and start the parallel/asynchronous operation
"""
#pdb.set_trace()

trigger_controller = RealtimeTrigger(rt_controller.cycle_time, rt_controller.asynchronous_step)

trigger_simulator  = RealtimeTrigger(rt_simulator.cycle_time , rt_simulator.asynchronous_step)

trigger_estimator  = RealtimeTrigger(rt_estimator.cycle_time , rt_estimator.asynchronous_step)

"""
Define the maximum number of optimization steps you want. do-mpc will until you manually stop it 
via the flags, or until max_iter is encountered, after which the plotting will be executed
"""
max_iter = 100
manual_stop = False
while rt_controller.iter_count < max_iter and manual_stop == False:
    print("Waiting on the main thread...")
    # TODO: read flags and update the manual stop
    time.sleep(10)


trigger_controller.stop()
trigger_simulator.stop()
trigger_estimator.stop()

"""
All OPCUA services should be terminated and the communications closed
"""
rt_simulator.stop()
rt_controller.stop()
rt_estimator.stop()
opc_server.stop()
del(opc_server)


"""
Final steps: saving the data (pickling) and some plotting
"""
# Store results for animated plotting and more
do_mpc.data.save_results([mpc, simulator], 'polyreactor_results')

# Initialize graphic:
graphics = do_mpc.graphics.Graphics()

fig, ax = plt.subplots(5, sharex=True)
plt.ion()
# Configure plot:
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[0])
graphics.add_line(var_type='_x', var_name='T_adiab', axis=ax[1])
graphics.add_line(var_type='_x', var_name='accum_monom', axis=ax[2])
graphics.add_line(var_type='_u', var_name='m_dot_f', axis=ax[3])
graphics.add_line(var_type='_u', var_name='T_in_M', axis=ax[4])
graphics.add_line(var_type='_u', var_name='T_in_EK', axis=ax[4])

ax[0].set_ylabel('T_R [K]')
ax[1].set_ylabel('T_adiab [K]')
ax[2].set_ylabel('acc. monom')
ax[3].set_ylabel('m_dot_f')
ax[4].set_ylabel('T_in_M [K]')
ax[4].set_ylabel('T_in_EK [K]')

fig.align_ylabels()
plt.ion()

simu_lines = graphics.plot_results(rt_simulator.data)
plt.show()
input('Press any key to exit.')
