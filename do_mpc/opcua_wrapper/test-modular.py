#%%
import sys
from casadi import *
import importlib
import time
import do_mpc
import numpy as np
from RTServer_modular import RTServer, RTBase, NamespaceEntry, Namespace, ServerOpts, ClientOpts
# importlib.reload(RTServer_modular)

def bio_model():
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

    mpc = do_mpc.controller.MPC(model)

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

    suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    mpc.set_param(nlpsol_opts = suppress_ipopt)

    mpc.setup()

    return model

def mpc(model):
    

    mpc = do_mpc.controller.MPC(model)

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

    suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    mpc.set_param(nlpsol_opts = suppress_ipopt)

    mpc.setup()

    return mpc

def sim(model):
    simulator = do_mpc.simulator.Simulator(model)

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

    return simulator

model = bio_model()
controller = mpc(model)
simulator = sim(model)
# Initial guess
X_s_0 = 1.0 # Concentration biomass [mol/l]
S_s_0 = 0.5 # Concentration substrate [mol/l]
P_s_0 = 0.0 # Concentration product [mol/l]
V_s_0 = 120.0 # Volume inside tank [m^3]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])
controller.x0 = x0
controller.set_initial_guess()
simulator.x0 = x0


#%% Server setup


# Defining the settings for the OPCUA server
server_opts = ServerOpts("Bio Reactor OPCUA",   
               "opc.tcp://localhost:4840/freeopcua/server/",  
               4840)                 

server_opts_2 = ServerOpts("Bio Reactor OPCUA",      
               "opc.tcp://localhost:4840/freeopcua/server/",  
               4841)   



client_opts_1 = ClientOpts("Bio Reactor OPCUA Client_1","opc.tcp://localhost:4840/freeopcua/server/",4840)
client_opts_2 = ClientOpts("Bio Reactor OPCUA Client_2","opc.tcp://localhost:4840/freeopcua/server/",4840)

Server = RTServer(server_opts)
rt_mpc = RTBase(controller, client_opts_1)
rt_sim = RTBase(simulator, client_opts_2)

Server.namespace_from_client(rt_mpc)
Server.namespace_from_client(rt_sim)

rt_mpc.set_write_tags(rt_mpc.def_namespace['u'])
rt_sim.set_write_tags(rt_sim.def_namespace['x'])
rt_mpc.set_read_tags(rt_sim.def_namespace['x'])
rt_sim.set_read_tags(rt_mpc.def_namespace['u'])
#%%
# Server.get_all_nodes()

#%%
# rt_mpc.set_default_read_ns()

#%%
# Server.get_all_nodes()
Server.start()
rt_mpc.connect()
rt_sim.connect()

rt_sim.init_server()
#%%
rt_mpc.async_step_start()
rt_sim.async_step_start()
#%%

#%%
# Server.stop()
# %%

for i in range(2):
    print({'u':rt_mpc.client.readData('ns=2;s=u[inp]'),
    'x':rt_mpc.client.readData('ns=3;s=x[X_s]')})
    time.sleep(3)

rt_mpc.async_step_stop()
rt_sim.async_step_stop()
rt_mpc.disconnect()
rt_sim.disconnect()
# %%

''' 
Diskussionspunkte mit Felix:
-Speichern von daten (SQL? - probleme mit async function, mpc.storesolution?)
-gucken was felix an der set_read_write() class auszusetzen hat bzw. woher weiß base was sie ist?
-Fehlermeldungen
-felix nach seiner meinung zur Leikonsache fragen
-wie soll die manuelle ns eingabe ablaufen und welchen zweck soll sie haben?
-wie sieht der do-mpc output für n-dim aus? np.array([?])

TODO:

'''