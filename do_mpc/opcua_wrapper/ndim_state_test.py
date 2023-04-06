#%%
import sys
from casadi import *
import importlib
import time
import do_mpc
import numpy as np
from RTServer_modular import RTServer, RTBase, NamespaceEntry, Namespace, ServerOpts, ClientOpts
#%%
def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Same example as shown in the Jupyter Notebooks.

    # Model variables:
    phi_1 = model.set_variable(var_type='_x', var_name='phi_1')
    phi_2 = model.set_variable(var_type='_x', var_name='phi_2')
    phi_3 = model.set_variable(var_type='_x', var_name='phi_3')

    phi = vertcat(phi_1, phi_2, phi_3)

    dphi = model.set_variable(var_type='_x', var_name='dphi', shape=(3,1))

    # Two states for the desired (set) motor position:
    phi_m_set = model.set_variable(var_type='_u', var_name='phi_m_set', shape=(2,1))

    # Two additional states for the true motor position:
    phi_m = model.set_variable(var_type='_x', var_name='phi_m', shape=(2,1))

    # Set point for the central mass:
    phi_2_set = model.set_variable(var_type='_tvp', var_name='phi_2_set')

    # Parameter for the MHE: Weighting of the arrival cost (parameters):
    # P_p = model.set_variable(var_type='_p', var_name='P_p')

    # Time-varying parameter for the MHE: Weighting of the measurements (tvp):
    # P_v = model.set_variable(var_type='_tvp', var_name='P_v', shape=(5, 5))

    # State measurements
    # phi_meas = model.set_meas('phi_1_meas', phi)

    # Input measurements
    # phi_m_set_meas = model.set_meas('phi_m_set_meas', phi_m_set)

    Theta_1 = model.set_variable('parameter', 'Theta_1')
    Theta_2 = model.set_variable('parameter', 'Theta_2')
    Theta_3 = model.set_variable('parameter', 'Theta_3')

    c = np.array([2.697,  2.66,  3.05, 2.86])*1e-3
    d = np.array([6.78,  8.01,  8.82])*1e-5


    model.set_rhs('phi_1', dphi[0])
    model.set_rhs('phi_2', dphi[1])
    model.set_rhs('phi_3', dphi[2])

    dphi_next = vertcat(
        -c[0]/Theta_1*(phi[0]-phi_m[0])-c[1]/Theta_1*(phi[0]-phi[1])-d[0]/Theta_1*dphi[0],
        -c[1]/Theta_2*(phi[1]-phi[0])-c[2]/Theta_2*(phi[1]-phi[2])-d[1]/Theta_2*dphi[1],
        -c[2]/Theta_3*(phi[2]-phi[1])-c[3]/Theta_3*(phi[2]-phi_m[1])-d[2]/Theta_3*dphi[2],
    )

    model.set_rhs('dphi', dphi_next)

    tau = 1e-2
    model.set_rhs('phi_m', 1/tau*(phi_m_set - phi_m))

    model.setup()

    return model


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 20,
        't_step': 0.1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }

    mpc.set_param(**setup_mpc)

    _x, _tvp  = model['x', 'tvp']

    lterm = (_x['phi_2'] - _tvp['phi_2_set'])**2
    mterm = DM(1)

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(phi_m_set=1e-2)


    # Create an interesting trajectory for the setpoint (_tvp)
    # by randomly choosing a new value or keeping the previous one.
    def random_setpoint(tvp0):
        # tvp_next = (0.5-np.random.rand(1))*np.pi
        # switch = np.random.rand() >= 0.95
        # tvp0 = (1-switch)*tvp0 + switch*tvp_next
        return tvp0

    np.random.seed(999)
    tvp_traj = [np.array([0])]
    for i in range(400):
        tvp_traj.append(random_setpoint(tvp_traj[i]))

    tvp_traj = np.concatenate(tvp_traj)

    # Create tvp_fun that takes element from that previously defined trajectory
    # depending on the current timestep.
    tvp_template = mpc.get_tvp_template()
    def tvp_fun(t_now):
        ind = int(t_now/setup_mpc['t_step'])
        tvp_template['_tvp', :-1] = vertsplit(tvp_traj[ind:ind+setup_mpc['n_horizon']])
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    inertia_mass_1 = 2.25*1e-4*np.array([1.,1.1])
    inertia_mass_2 = 2.25*1e-4*np.array([1.,])
    inertia_mass_3 = 2.25*1e-4*np.array([1.])

    mpc.set_uncertainty_values(
        Theta_1 = inertia_mass_1,
        Theta_2 = inertia_mass_2,
        Theta_3 = inertia_mass_3,)

    mpc.bounds['lower','_u','phi_m_set'] = -5
    mpc.bounds['upper','_u','phi_m_set'] = 5

    suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
    mpc.set_param(nlpsol_opts = suppress_ipopt)
    mpc.setup()

    return mpc

def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)


    simulator.set_param(t_step = 0.1)

    p_template = simulator.get_p_template()
    def p_fun(t_now):
        p_template['Theta_1'] = 2.25e-4
        p_template['Theta_2'] = 2.25e-4
        p_template['Theta_3'] = 2.25e-4
        return p_template
    simulator.set_p_fun(p_fun)

    # The timevarying paramters have no effect on the simulator (they are only part of the cost function).
    # We simply use the default values:
    tvp_template = simulator.get_tvp_template()
    def tvp_fun(t_now):
        return tvp_template

    simulator.set_tvp_fun(tvp_fun)


    simulator.setup()

    return simulator
# %%

model = template_model()
controller  = template_mpc(model)
simulator = template_simulator(model)

#%%
np.random.seed(99)
x0 = np.random.rand(model.n_x)-0.5
controller.x0 = x0
simulator.x0 = x0
controller.set_initial_guess()




# Defining the settings for the OPCUA server
server_opts = ServerOpts("Bio Reactor OPCUA",   
               "opc.tcp://localhost:4840/freeopcua/server/",  
               4840)                 

client_opts_1 = ClientOpts("Bio Reactor OPCUA Client_1","opc.tcp://localhost:4840/freeopcua/server/",4840)
client_opts_2 = ClientOpts("Bio Reactor OPCUA Client_2","opc.tcp://localhost:4840/freeopcua/server/",4840)

Server = RTServer(server_opts)
rt_mpc = RTBase(controller, client_opts_1)
rt_sim = RTBase(simulator, client_opts_2)

Server.namespace_from_client(rt_mpc)
Server.namespace_from_client(rt_sim)

#%%
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


rt_sim.write_to_tags(simulator.x0)
rt_mpc.write_to_tags(controller.u0)
#%%
rt_mpc.async_step_start()
rt_sim.async_step_start()
#%%

#%%
# Server.stop()
# %%
state_list = []
input_list= []
for i in range(7*60):
    print({'u':rt_sim.read_from_tags(),
    'x':rt_mpc.read_from_tags()})
    state_list.append(rt_mpc.read_from_tags())
    input_list.append(rt_sim.read_from_tags())
    time.sleep(3)

rt_mpc.async_step_stop()
rt_sim.async_step_stop()
rt_mpc.disconnect()
rt_sim.disconnect()
time.sleep(3)
Server.stop()
#%%
