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
from template_opcua import template_opcua

from opcmodules import Server, Client
from opcmodules import RealtimeSimulator, RealtimeController, RealtimeEstimator
from opcmodules import RealtimeTrigger

"""
User settings
"""
store_data   = False
plot_results = True


model = template_model()

opc_server, opc_opts = template_opcua(model)
    
rt_simulator = template_simulator(model,opc_opts)

rt_estimator = template_estimator(model, opc_opts)

rt_controller = template_mpc(model,opc_opts)

"""
Initialization and preparation of the server data base
"""

# The user is an object that lets the main thread access the OPCUA server for status and flag checks
opc_opts['_opc_opts']['_client_type'] = 'ManualUser'
user = Client(opc_opts['_opc_opts'])
user.connect()
# Start all modules if the manual mode is enabled and avoid delays
if opc_opts['_user_controlled']: 
    user.updateSwitches(pos = -1, switchVal=[1,1,1])

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
# Step 2: initialize the estimator part (if present)
rt_estimator.init_server(x0.cat.toarray().tolist())
rt_estimator.set_initial_state(x0, reset_history=True)
# Step 3: only now can the optimizer be initialized, and a first optimization can be executed
time.sleep((rt_simulator.cycle_time+rt_estimator.cycle_time)/2)
rt_controller.set_initial_state(x0, reset_history=True)
rt_controller.init_server(rt_controller._u0.cat.toarray().tolist())

# Step 4: the cyclical operation can be safely started now
"""
Define triggers for each of the modules and start the parallel/asynchronous operation
"""
trigger_simulator  = RealtimeTrigger(rt_simulator.cycle_time , rt_simulator.asynchronous_step)

trigger_estimator  = RealtimeTrigger(rt_estimator.cycle_time , rt_estimator.asynchronous_step)

trigger_controller = RealtimeTrigger(rt_controller.cycle_time, rt_controller.asynchronous_step)

"""
The real-time do-mpc will keep running until you manually stop it via the flags (use an OPCUA Client to set the flags).
Alternatively, use the routine below to check when the maximum nr of iterations is reached and stop the real-time modules.
"""
max_iter = 10
manual_stop = False

while rt_controller.iter_count < max_iter and manual_stop == False:
    # The code below is executed on the main thread (e.g the Ipython console you're using to start do-mpc)
    print("Waiting on the main thread...Checking flags...Executing your main code...")
    
    if user.checkFlags(pos=0) == 1:
        print("The controller has failed! Better take backup action ...")
    if user.checkFlags(pos=0) == 0: print("All systems OK @ ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
    
    # Checking the status of the modules
    switches = user.checkSwitches()
    print("The controller is:", 'ON' if switches[0] else 'OFF')
    print("The simulator  is:", 'ON' if switches[1] else 'OFF')
    print("The estimator  is:", 'ON' if switches[2] else 'OFF')
    
    # Check the 5th flag and stop all modules at once if the user has raised the flag
    # Alternatively, the user can individually stop modules by setting the switch to 0
    if user.checkFlags(pos=4) == 1: 
        user.updateSwitches(pos = -1, switchVal=[0,0,0])
        manual_stop = True
    # The main thread sleeps for 7 seconds and repeats
    time.sleep(10)

# Once the main thread reaches this point, all real-time modules will be stopped
trigger_controller.stop()
trigger_simulator.stop()
trigger_estimator.stop()

"""
All OPCUA services should be terminated and the communications closed, to prevent Python errors
"""
rt_simulator.stop()
rt_controller.stop()
rt_estimator.stop()
opc_server.stop()
del(opc_server)


"""
Final steps: saving the data (pickling) and some plotting
"""
if store_data:
    # Store results for animated plotting and more
    do_mpc.data.save_results([rt_controller, rt_simulator, rt_estimator], 'polyreactor_results')

if plot_results:
    # Initialize graphic:
    mpc_graphics = do_mpc.graphics.Graphics(rt_controller.data)
    est_graphics = do_mpc.graphics.Graphics(rt_estimator.data)
    fig, ax = plt.subplots(6, sharex=True)
    plt.ion()
    # Configure plot:
    mpc_graphics.add_line(var_type='_x', var_name='T_R', axis=ax[0])
    mpc_graphics.add_line(var_type='_x', var_name='T_adiab', axis=ax[1])
    mpc_graphics.add_line(var_type='_x', var_name='accum_monom', axis=ax[2])
    mpc_graphics.add_line(var_type='_x', var_name='m_P', axis=ax[3])
    mpc_graphics.add_line(var_type='_u', var_name='m_dot_f', axis=ax[4])
    mpc_graphics.add_line(var_type='_u', var_name='T_in_M', axis=ax[5])
    mpc_graphics.add_line(var_type='_u', var_name='T_in_EK', axis=ax[5])
    # For comparisson also plot some estimated  data
    est_graphics.add_line(var_type='_x', var_name='T_R', axis=ax[0], linestyle = '-.', color='#1f77b4')
    est_graphics.add_line(var_type='_x', var_name='T_adiab', axis=ax[1], linestyle = '-.', color='#1f77b4')
    est_graphics.add_line(var_type='_x', var_name='m_P', axis=ax[3], linestyle = '-.', color='#ff7f0e')
    
    # Adding some labels for easy data reads
    label_lines = mpc_graphics.result_lines['_x', 'T_R']+est_graphics.result_lines['_x', 'T_R']
    ax[0].legend(label_lines, ['Real $T_R$', 'Estimated $T_R$'])
    
    label_lines = mpc_graphics.result_lines['_x', 'T_adiab']+est_graphics.result_lines['_x', 'T_adiab']
    ax[1].legend(label_lines, ['Real $T_{adiab}$', 'Estimated $T_{adiab}$'])
    
    label_lines = mpc_graphics.result_lines['_x', 'm_P']+est_graphics.result_lines['_x', 'm_P']
    ax[3].legend(label_lines, ['Real $m_P$', 'Estimated $m_P$'])
    
    label_lines = mpc_graphics.result_lines['_u', 'T_in_M']+mpc_graphics.result_lines['_u', 'T_in_EK']
    ax[5].legend(label_lines, ['T_in_M', 'T_in_EK'])
    
    ax[0].set_ylabel('T_R [K]')
    ax[1].set_ylabel('T_adiab [K]')
    ax[2].set_ylabel('Acc. monomer')
    ax[3].set_ylabel('Mass Polymer')
    ax[4].set_ylabel('m_dot_f')
    ax[5].set_ylabel('T_in [K]')
    ax[5].set_xlabel('Time [h]')
    
    fig.align_ylabels()
    plt.ion()
    
    simu_lines = mpc_graphics.plot_results(t_ind=rt_controller.iter_count)
    plt.show()

#input('Press any key to exit.')
