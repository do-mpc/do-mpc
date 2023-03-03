# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 13:58:17 2023

@author: User
"""

import numpy as np
import do_mpc
from casadi import *
import time
import opcua_wrapper
from casadi.tools import *
import pdb

#%% Model

"""
--------------------------------------------------------------------------
template_model: Variables / RHS / AUX
--------------------------------------------------------------------------
"""
model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)

# Certain parameters
R           = 8.314                 #gas constant
T_F         = 25 + 273.15   #feed temperature
E_a         = 8500.0                #activation energy
delH_R      = 950.0*1.00    #sp reaction enthalpy
A_tank      = 65.0                  #area heat exchanger surface jacket 65

k_0         = 7.0*1.00              #sp reaction rate
k_U2        = 32.0                  #reaction parameter 1
k_U1        = 4.0                   #reaction parameter 2
w_WF        = .333                  #mass fraction water in feed
w_AF        = .667                  #mass fraction of A in feed

m_M_KW      = 5000.0                #mass of coolant in jacket
fm_M_KW     = 300000.0              #coolant flow in jacket 300000;
m_AWT_KW    = 1000.0                #mass of coolant in EHE
fm_AWT_KW   = 100000.0              #coolant flow in EHE
m_AWT       = 200.0                 #mass of product in EHE
fm_AWT      = 20000.0               #product flow in EHE
m_S         = 39000.0               #mass of reactor steel

c_pW        = 4.2                   #sp heat cap coolant
c_pS        = .47                   #sp heat cap steel
c_pF        = 3.0                   #sp heat cap feed
c_pR        = 5.0                   #sp heat cap reactor contents

k_WS        = 17280.0               #heat transfer coeff water-steel
k_AS        = 3600.0                #heat transfer coeff monomer-steel
k_PS        = 360.0                 #heat transfer coeff product-steel

alfa        = 5*20e4*3.6

p_1         = 1.0

# Uncertain parameters:
delH_R = model.set_variable('_p', 'delH_R')
k_0 =    model.set_variable('_p', 'k_0')


# States struct (optimization variables):
m_W =         model.set_variable('_x', 'm_W')
m_A =         model.set_variable('_x', 'm_A')
m_P =         model.set_variable('_x', 'm_P')
T_R =         model.set_variable('_x', 'T_R')
T_S =         model.set_variable('_x', 'T_S')
Tout_M =      model.set_variable('_x', 'Tout_M')
T_EK =        model.set_variable('_x', 'T_EK')
Tout_AWT =    model.set_variable('_x', 'Tout_AWT')
accum_monom = model.set_variable('_x', 'accum_monom')
T_adiab =     model.set_variable('_x', 'T_adiab')

# Input struct (optimization variables):
m_dot_f = model.set_variable('_u', 'm_dot_f')
T_in_M =  model.set_variable('_u', 'T_in_M')
T_in_EK = model.set_variable('_u', 'T_in_EK')

# algebraic equations
U_m    = m_P / (m_A + m_P)
m_ges  = m_W + m_A + m_P
k_R1   = k_0 * exp(- E_a/(R*T_R)) * ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
k_R2   = k_0 * exp(- E_a/(R*T_EK))* ((k_U1 * (1 - U_m)) + (k_U2 * U_m))
k_K    = ((m_W / m_ges) * k_WS) + ((m_A/m_ges) * k_AS) + ((m_P/m_ges) * k_PS)

# Differential equations
dot_m_W = m_dot_f * w_WF
model.set_rhs('m_W', dot_m_W)
dot_m_A = (m_dot_f * w_AF) - (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) - (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
model.set_rhs('m_A', dot_m_A)
dot_m_P = (k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT)
model.set_rhs('m_P', dot_m_P)

dot_T_R = 1./(c_pR * m_ges)   * ((m_dot_f * c_pF * (T_F - T_R)) - (k_K *A_tank* (T_R - T_S)) - (fm_AWT * c_pR * (T_R - T_EK)) + (delH_R * k_R1 * (m_A-((m_A*m_AWT)/(m_W+m_A+m_P)))))
model.set_rhs('T_R', dot_T_R)
model.set_rhs('T_S', 1./(c_pS * m_S)     * ((k_K *A_tank* (T_R - T_S)) - (k_K *A_tank* (T_S - Tout_M))))
model.set_rhs('Tout_M', 1./(c_pW * m_M_KW)  * ((fm_M_KW * c_pW * (T_in_M - Tout_M)) + (k_K *A_tank* (T_S - Tout_M))))
model.set_rhs('T_EK', 1./(c_pR * m_AWT)   * ((fm_AWT * c_pR * (T_R - T_EK)) - (alfa * (T_EK - Tout_AWT)) + (p_1 * k_R2 * (m_A/m_ges) * m_AWT * delH_R)))
model.set_rhs('Tout_AWT', 1./(c_pW * m_AWT_KW)* ((fm_AWT_KW * c_pW * (T_in_EK - Tout_AWT)) - (alfa * (Tout_AWT - T_EK))))
model.set_rhs('accum_monom', m_dot_f)
model.set_rhs('T_adiab', delH_R/(m_ges*c_pR)*dot_m_A-(dot_m_A+dot_m_W+dot_m_P)*(m_A*delH_R/(m_ges*m_ges*c_pR))+dot_T_R)
    
model.setup()




#%% Server setup


# Defining the settings for the OPCUA server
server_opts = {"_model":model,                   # must be the model used to define the controller
               "_name":"Poly Reactor OPCUA",     # give the server whatever name you prefer
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


#%% Controller setup
control_opts = {}
control_opts['_opc_opts'] = client_opts
control_opts['_cycle_time'] = 10.0
control_opts['_output_feedback'] = True
control_opts['_user_controlled'] = False
control_opts['_opc_opts']['_client_type'] = 'controller'

mpc = opcua_wrapper.RealtimeController(model,control_opts)

#%% MPC setup
"""
--------------------------------------------------------------------------
template_mpc: tuning parameters
--------------------------------------------------------------------------
"""


setup_mpc = {
    'n_horizon': 20,
    'n_robust': 0,
    'open_loop': 0,
    't_step': 50.0/3600.0,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

_x = model.x
mterm = - _x['m_P'] # terminal cost
lterm = - _x['m_P'] # stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(m_dot_f=0.002, T_in_M=0.004, T_in_EK=0.002) # penalty on control input changes

# auxiliary term
temp_range = 2.0



# lower bound states
mpc.bounds['lower','_x','m_W'] = 0.0
mpc.bounds['lower','_x','m_A'] = 0.0
mpc.bounds['lower','_x','m_P'] = 26.0

mpc.bounds['lower','_x','T_R'] = 363.15 - temp_range
mpc.bounds['lower','_x','T_S'] = 298.0
mpc.bounds['lower','_x','Tout_M'] = 298.0
mpc.bounds['lower','_x','T_EK'] = 288.0
mpc.bounds['lower','_x','Tout_AWT'] = 288.0
mpc.bounds['lower','_x','accum_monom'] = 0.0

# upper bound states
mpc.bounds['upper','_x','T_S'] = 400.0
mpc.bounds['upper','_x','Tout_M'] = 400.0
mpc.bounds['upper','_x','T_EK'] = 400.0
mpc.bounds['upper','_x','Tout_AWT'] = 400.0
mpc.bounds['upper','_x','accum_monom'] = 30000.0
mpc.bounds['upper','_x','T_adiab'] = 382.15


mpc.set_nl_cons('T_R_UB', _x['T_R'], ub=363.15+temp_range, soft_constraint=True, penalty_term_cons=1e4)

# lower bound inputs
mpc.bounds['lower','_u','m_dot_f'] = 0.0
mpc.bounds['lower','_u','T_in_M'] = 333.15
mpc.bounds['lower','_u','T_in_EK'] = 333.15

# upper bound inputs
mpc.bounds['upper','_u','m_dot_f'] = 3.0e4
mpc.bounds['upper','_u','T_in_M'] = 373.15
mpc.bounds['upper','_u','T_in_EK'] = 373.15

# states
mpc.scaling['_x','m_W'] = 10
mpc.scaling['_x','m_A'] = 10
mpc.scaling['_x','m_P'] = 10
mpc.scaling['_x','accum_monom'] = 10

# control inputs
mpc.scaling['_u','m_dot_f'] = 100

delH_R_var = np.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
k_0_var = np.array([7.0 * 1.00, 7.0 * 1.30, 7.0 * 0.70])

mpc.set_uncertainty_values(delH_R = delH_R_var, k_0 = k_0_var)

mpc.setup()

#%% Simulator setup
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
    't_step': 50.0/3600.0
}

simulator.set_param(**params_simulator)

p_num = simulator.get_p_template()
tvp_num = simulator.get_tvp_template()

# uncertain parameters
p_num['delH_R'] = 950 * np.random.uniform(0.75,1.25)
p_num['k_0'] = 7 * np.random.uniform(0.75*1.25)
def p_fun(t_now):
    return p_num
simulator.set_p_fun(p_fun)

simulator.setup()

#%% Estimator setup

est_opts = {}
est_opts['_opc_opts'] = client_opts
est_opts['_cycle_time'] = 4
est_opts['_user_controlled'] = False
est_opts['_opc_opts']['_client_type'] = 'estimator'
est_opts['_opc_opts']['_output_feedback'] = False

estimator = opcua_wrapper.RealtimeEstimator('state-feedback',model,est_opts)

#%% Initial guesses

# Set the initial state of the controller and simulator:
# assume nominal values of uncertain parameters as initial guess
delH_R_real = 950.0
c_pR = 5.0

# x0 is a property of the simulator - we obtain it and set values.
x0 = simulator.x0

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

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

#%% Initiialize server

# Step 1: initilize the simulator part
simulator.init_server() #soll keinen input brauchen weil das ja klar ist...

# Step 2: initialize the estimator part (if present)
estimator.init_server()

# Step 3: only now can the optimizer be initialized, and a first optimization can be executed
time.sleep((simulator.cycle_time+estimator.cycle_time)/2)
mpc.init_server()


#%% Realtime trigger
"""
Define triggers for each of the modules and start the parallel/asynchronous operation
"""
trigger_simulator  = opcua_wrapper.RealtimeTrigger(simulator.cycle_time , simulator.asynchronous_step)

trigger_estimator  = opcua_wrapper.RealtimeTrigger(estimator.cycle_time , estimator.asynchronous_step)

trigger_controller = opcua_wrapper.RealtimeTrigger(mpc.cycle_time, mpc.asynchronous_step)
#leichte möglichkeit den server auszulesen


time.sleep(60)

#%% Stop trigger
trigger_controller.stop()
trigger_simulator.stop()
trigger_estimator.stop()
#%% stop server and clients

mpc.stop()
simulator.stop()
estimator.stop()
opc_server.stop()