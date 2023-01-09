# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:36:19 2022

@author: Felix Brabender
"""

import numpy as np
import do_mpc
from casadi import *
import time
import opcua_wrapper


#%% Setup model

model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)
# States struct (optimization variables):
X_s = model.set_variable('_x',  'X_s')
S_s = model.set_variable('_x',  'S_s')
P_s = model.set_variable('_x',  'P_s')
V_s = model.set_variable('_x',  'V_s')
# Input struct (optimization variables):
inp = model.set_variable('_u',  'inp')
# Certain parameters
mu_m  = 0.02
K_m   = 0.05
K_i   = 5.0
v_par = 0.004
Y_p   = 1.2

# Uncertain parameters:
Y_x  = model.set_variable('_p',  'Y_x')
S_in = model.set_variable('_p', 'S_in')
# Auxiliary term
mu_S = mu_m*S_s/(K_m+S_s+(S_s**2/K_i))

# Differential equations
model.set_rhs('X_s', mu_S*X_s - inp/V_s*X_s)
model.set_rhs('S_s', -mu_S*X_s/Y_x - v_par*X_s/Y_p + inp/V_s*(S_in-S_s))
model.set_rhs('P_s', v_par*X_s - inp/V_s*P_s)
model.set_rhs('V_s', inp)
# Build the model
model.setup()

#%% Server setup


# Defining the settings for the OPCUA server
server_opts = {"_model":model,                   # must be the model used to define the controller
               "_name":"Bio Reactor OPCUA",     # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4840,                    # does not need changing
               "_server_type": "with_estimator", # to use with either SFB or EKF estimators or select "basic" for no estimates
               "_store_params": True,            # should always be set to True to spare yourself the headaches
               "_store_predictions": False,      # set to True only if you plan to do fancy things with the predictions
               "_with_db": True,                 # set to True if you plan to use the SQL database during/after the runtime 
               "_n_steps_pred": 20}
# Create your OPUA server
opc_server = opcua_wrapper.Server(server_opts)


#%%Start server
# The server can only be started if it hasn't been already created
if opc_server.created == True and opc_server.running == False: opc_server.start()


#%% Client setup

client_opts = {"_address":"opc.tcp://localhost:4840/freeopcua/server/", # the basic implementation, can remain as is
               "_port": 4840,                              # should remain as is
               "_client_type": "ManualUser",               # simulator, estimator, user
               "_namespace": opc_server.namespace}         # must match the server, therefore simply copy the namespace 

# The user is an object that lets the main thread access the OPCUA server for status and flag checks
client_opts['_client_type'] = 'ManualUser'
user = opcua_wrapper.Client(client_opts)
#Connect to the server
user.connect()

# Example 1: read all the operation flags
my_flags   = user.checkFlags()
# or read just one of the 5 flags, at position 0
one_flag   = user.checkFlags(pos = 0)
# Example 2: switch on/off the MPC modules
# update_res = user.updateSwitches(switchVal=[1,1,1])

#%% Controller setup
control_opts = {}
control_opts['_opc_opts'] = client_opts
control_opts['_cycle_time'] = 10.0
control_opts['_output_feedback'] = True
control_opts['_user_controlled'] = False
control_opts['_opc_opts']['_client_type'] = 'controller'

mpc = opcua_wrapper.RealtimeController(model,control_opts)



#%%
setup_mpc = {
    'n_horizon': 20,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 1.0,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

mterm = -model.x['P_s'] # terminal cost
lterm = -model.x['P_s'] # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(inp=1.0) # penalty on input changes

# lower bounds of the states
mpc.bounds['lower', '_x', 'X_s'] = 0.0
mpc.bounds['lower', '_x', 'S_s'] = -0.01
mpc.bounds['lower', '_x', 'P_s'] = 0.0
mpc.bounds['lower', '_x', 'V_s'] = 0.0

# upper bounds of the states
mpc.bounds['upper', '_x','X_s'] = 3.7
mpc.bounds['upper', '_x','P_s'] = 3.0

# upper and lower bounds of the control input
mpc.bounds['lower','_u','inp'] = 0.0
mpc.bounds['upper','_u','inp'] = 0.2

Y_x_values = np.array([0.5, 0.4, 0.3])
S_in_values = np.array([200.0, 220.0, 180.0])

mpc.set_uncertainty_values(Y_x = Y_x_values, S_in = S_in_values)

mpc.setup()

#%%
sim_opts = {}
sim_opts['_opc_opts'] = client_opts
sim_opts['_cycle_time'] = 2
sim_opts['_user_controlled'] = False
sim_opts['_opc_opts']['_client_type'] = 'simulator'

simulator = opcua_wrapper.RealtimeSimulator(model,sim_opts)

params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-10,
    'reltol': 1e-10,
    't_step': 1.0
}

simulator.set_param(**params_simulator)

p_num = simulator.get_p_template()

p_num['Y_x'] = 0.4
p_num['S_in'] = 200.0

# function definition
def p_fun(t_now):
    return p_num

# Set the user-defined function above as the function for the realization of the uncertain parameters
simulator.set_p_fun(p_fun)

simulator.setup()

#%%

est_opts = {}
est_opts['_opc_opts'] = client_opts
est_opts['_cycle_time'] = 4
est_opts['_user_controlled'] = False
est_opts['_opc_opts']['_client_type'] = 'estimator'
est_opts['_opc_opts']['_output_feedback'] = False

estimator = opcua_wrapper.RealtimeEstimator('state-feedback',model,est_opts)

#%%

# Initial state
X_s_0 = 1.0 # Concentration biomass [mol/l]
S_s_0 = 0.5 # Concentration substrate [mol/l]
P_s_0 = 0.0 # Concentration product [mol/l]
V_s_0 = 120.0 # Volume inside tank [m^3]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

#%%

# Step 1: initilize the simulator part
simulator.x0 = x0
simulator.init_server() #soll keinen input brauchen weil das ja klar ist...

# Step 2: initialize the estimator part (if present)
estimator.x0 = x0
estimator.init_server()

# Step 3: only now can the optimizer be initialized, and a first optimization can be executed
time.sleep((simulator.cycle_time+estimator.cycle_time)/2)
mpc.x0 = x0
mpc.init_server()


#%%
"""
Define triggers for each of the modules and start the parallel/asynchronous operation
"""
trigger_simulator  = opcua_wrapper.RealtimeTrigger(simulator.cycle_time , simulator.asynchronous_step)

trigger_estimator  = opcua_wrapper.RealtimeTrigger(estimator.cycle_time , estimator.asynchronous_step)

trigger_controller = opcua_wrapper.RealtimeTrigger(mpc.cycle_time, mpc.asynchronous_step)
#leichte möglichkeit den server auszulesen


time.sleep(60)

#%%
trigger_controller.stop()
trigger_simulator.stop()
trigger_estimator.stop()
#%%
# opc_server.output()
# opc_server.export_xml_by_ns('C:/Users/User/Desktop/test.xml')


user.disconnect()
mpc.stop()
simulator.stop()
estimator.stop()
opc_server.stop()

