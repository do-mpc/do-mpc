#%%
import sys
from casadi import *
import importlib
import time
import do_mpc
import numpy as np
from opcua_wrapper import *
from multiprocessing import Process, Queue

#%% Create do-mpc classes

def bio_model():
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)

    # States struct (optimization variables):
    X_s = model.set_variable('_x',  'X_s', )
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
    't_step': 0.01,
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
        't_step': 0.01
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

#%% Setup server and dummy clients to get namespaces automatically

server_opts = ServerOpts("Bio Reactor OPCUA",   
               "opc.tcp://localhost:4840/freeopcua/server/",  
               4840)                      

Server = RTServer(server_opts)
rt_mpc = RTBase(controller, ClientOpts("Bio Reactor OPCUA Client_2","opc.tcp://localhost:4840/freeopcua/server/",4840))
rt_sim = RTBase(simulator, ClientOpts("Bio Reactor OPCUA Client_1","opc.tcp://localhost:4840/freeopcua/server/",4840))

Server.namespace_from_client(rt_mpc)
Server.namespace_from_client(rt_sim)

mpc_write_tags = rt_mpc.def_namespace['u']
sim_write_tags  = rt_sim.def_namespace['x']
mpc_read_tags = rt_sim.def_namespace['x']
sim_read_tags = rt_mpc.def_namespace['u']


#%% Client setup for multiprocessing
def send_mpc_to_process(read_tags, write_tags):
    model_p = bio_model()
    controller_p = mpc(model_p)
    controller_p.x0 = np.array([1.0, 0.5, 0.0, 120.0])
    controller_p.set_initial_guess()

    rt_mpc_p = RTBase(controller_p, ClientOpts("Bio Reactor OPCUA Client_1","opc.tcp://localhost:4840/freeopcua/server/",4840))
    rt_mpc_p.set_write_tags(write_tags)
    rt_mpc_p.set_read_tags(read_tags)
    rt_mpc_p.connect()
    rt_mpc_p.write_to_tags(controller_p.u0)
    rt_mpc_p.async_step_start()
    time.sleep(20)
    rt_mpc_p.async_step_stop()
    rt_mpc_p.disconnect()


def send_sim_to_process(q, read_tags, write_tags):
    model_p = bio_model()
    simulator_p = sim(model_p)
    simulator_p.x0 = np.array([1.0, 0.5, 0.0, 120.0])

    rt_sim_p = RTBase(simulator_p, ClientOpts("Bio Reactor OPCUA Observer","opc.tcp://localhost:4840/freeopcua/server/",4840))
    rt_sim_p.set_write_tags(write_tags)
    rt_sim_p.set_read_tags(read_tags)
    rt_sim_p.connect()
    rt_sim_p.write_to_tags(simulator_p.x0)
    rt_sim_p.async_step_start()
    time.sleep(20)
    rt_sim_p.async_step_stop()
    q.put(rt_sim_p.do_mpc_object.data)
    rt_sim_p.disconnect()
    


# Observer client for main process
obs_client = RTClient(ClientOpts("Bio Reactor OPCUA Observer","opc.tcp://localhost:4840/freeopcua/server/",4840), [])




#%% Run all processes

if __name__ == '__main__':

    q = Queue()

    Server.start()
    obs_client.connect()

    sim_process = Process(target=send_sim_to_process, args=(q, sim_read_tags, sim_write_tags))
    sim_process.start()

    

    mpc_process = Process(target=send_mpc_to_process, args=(mpc_read_tags, mpc_write_tags))
    mpc_process.start()


    time.sleep(10)

    for i in range(10):
        mpc_test = obs_client.readData(mpc_write_tags[0])
        sim_test = np.array([obs_client.readData(sim_write_tags[0]), obs_client.readData(sim_write_tags[1]), obs_client.readData(sim_write_tags[2]), obs_client.readData(sim_write_tags[3])])

        print(f'controller write (u): {mpc_test} simulator write (X0-X3): {sim_test}')
        time.sleep(3)
    
    
    res = q.get()
    
    sim_process.join(timeout=20)
    
    mpc_process.join(timeout=20)
    Server.stop()