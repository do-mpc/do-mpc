# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 12:23:15 2023

@author: User
"""

#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2020 Sergio Lucia, Felix Riedl, Alexandru Tatulea-Codrean
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

#%%
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List
from threading import Timer, Thread
import do_mpc
try:
    import asyncua.sync as opcua
except ImportError:
    raise ImportError("The asyncua library is not installed. Please install it and try again.")
        
from asyncua.server.history_sql import HistorySQLite
#%%

class RTServer:

    def __init__(self, opts):
       
        # The basic OPCUA server definition contains a name, address and a port numer
        # The user can decide if they want to activate the SQL database option (_with_db=TRUE)
        self.name    = opts.name
        self.address = opts.address
        self.port    = opts.port
        self.with_db = opts.db
        self.namespace = []

        # Try to create the server
        try:
            self.opcua_server = opcua.Server()
            print("A server named - {} - was created".format(self.name))
        except RuntimeError:
            self.created = False
            print("Server could not be created. Check your opcua module installation!")
            return False

            
        
        # Mark the server as created & not yet running
        self.created = True
        self.running = False
        
        # Initiate SQL database
        if self.with_db == True:
            self.opcua_server.aio_obj.iserver.history_manager.set_storage(HistorySQLite('node_history.sql'))
        
    def namespace_from_client(self, client):
        # Obtain all namespaces saved by the client
        self.namespace += client.return_namespace()
        
        # Create the obtained namespaces on the server, if they do not already exist
        for key_0 in self.namespace:
            object_dict = {}
            if key_0._namespace_index == None:
                idx = self.opcua_server.register_namespace(key_0.namespace_name)
            else:
                print(f"The namespace {key_0.namespace_name} was already registered on this server.")
                continue
            

            for key_1 in key_0.entry_list:
                if  key_1.dim > 0:
                    if key_1.objectnode not in object_dict.keys():
                        object = self.opcua_server.nodes.objects.add_object(idx, key_1.objectnode)
                        placeholder = [0.0] * key_1.dim
                        variable_name_ = key_1.objectnode + '[' + key_1.variable + ']'
                        datavector = object.add_variable(opcua.ua.NodeId(variable_name_, idx), variable_name_, placeholder)
                        datavector.set_writable()
                        object_dict[key_1.objectnode] = object
                        key_1.variable = variable_name_

                    else:
                        placeholder = [0.0] * key_1.dim
                        variable_name_ = key_1.objectnode + '[' + key_1.variable + ']'
                        datavector = object_dict[key_1.objectnode].add_variable(opcua.ua.NodeId(variable_name_, idx), variable_name_, placeholder)
                        datavector.set_writable()
                        key_1.variable = variable_name_
                        
                else:
                    continue

            key_0._namespace_index = idx
            client.client.add_namespace_url(idx)
            
        self.opcua_server.get_namespace_array()
        print(f"The following namespaces are registered: {self.opcua_server.get_namespace_array()[2:]}")
    

    # Get a list with all node ID's for the SQL database    
    def get_all_nodes(self):        
        node_list = []
        for ns_entry in self.namespace:
            for ns_object in ns_entry.entry_list:
                id = ns_object.get_node_id(ns_entry._namespace_index)
                node_list.append(self.opcua_server.get_node(id))
        return node_list
    
    
    # Start server
    def start(self):
        try:
            self.opcua_server.start()
            if self.with_db == True:
                for it in self.get_all_nodes():
                    try:
                        self.opcua_server.aio_obj.historize_node_data_change(it,count=1e6)
                    except:
                        print("SQL database error, the following node can not be historized:\n", it) 
            self.running = True            
            return True
        except RuntimeError as err:
            print("The server "+ self.name +" could not be started, returned error message :\n", err)
            return False
    
        
    # Stop server
    def stop(self):
        try:
            self.opcua_server.stop()

            print("The server  "+ self.name +" was stopped successfully @ ",time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            self.running = False
            return True
        except RuntimeError as err:
            print("The server could not be stopped, returned error message :\n", err)
            return False
#%%
class RTClient:

    def __init__(self, opts, write_namespace):       
        # Information for connection to server
        self.server_address = opts.address
        self.port           = opts.port
        self.name           = opts.name
        
        # Information relevant for the client specific namespace
        self.namespace_list = [write_namespace]

        # Create Client and check if server is responding
        try:
            self.opcua_client = opcua.Client(self.server_address)
            print("A client named -", self.name, "- was created")
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The connection to the server could not be established\n", self.server_address, "is not responding")

    def return_namespace(self):        
        # Returns all registered namespaces
            return self.namespace_list
    
    def register_namespace_from_client(self, client):
        # Register namspaces from another client
        self.namespace_list.append(client.return_namespace()[0])
    
    def add_namespace_url(self, url):
        # Method used by server to mark namespaces with the corresponding url
        for item in self.namespace_list:
            item._namespace_index = url




    def connect(self):
        # This function implements (re)connection to the designated server
        try:
            self.opcua_client.connect()
            print("The -", self.name, "- has just connected to ",self.server_address)
            self.connected = True
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The connection to the server could not be established\n", self.server_address,
                  " is not responding")

    def disconnect(self):
        self.opcua_client.disconnect()
        self.connected = False
        print("A client of type", self.name,"disconnected from server",self.server_address)
        

    def writeData(self, dataVal, tag):

        assert type(dataVal) == list, "The data you provided is not arranged as a list. See the instructions for passing data to the server."
        
        try:
            wr_result = self.opcua_client.get_node(tag).set_value(dataVal)
        except ConnectionRefusedError:
            print("Write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return False
        
        return wr_result

    def readData(self, tag):
           
        try:
            dataVal = self.opcua_client.get_node(tag).get_value()
        except ConnectionRefusedError:
            print("A read operation by:", self.type, "failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        return dataVal
    



#%%        

class RTBase:

    def __init__(self, do_mpc_object, clientOpts, m_name):
        self.do_mpc_object = do_mpc_object
        self.def_namespace = namespace_from_model.detailed(self.do_mpc_object.model, model_name=m_name)
        self.cycle_time = do_mpc_object.t_step
        self.client = RTClient(clientOpts, self.def_namespace)
        self.tagout = []
        self.tagin = []
        self.init_server_tags = []
        self.is_running = False
        self.new_init = True


    def connect(self):
        try:
            self.client.connect()
            print("The real-time controller connected to the server")

            if self.writevariable == 'x':
                self.write_current(np.array(vertcat(self.do_mpc_object.x0)))

        except RuntimeError:
            self.enabled = False
            print("The real-time controller could not connect to the server. Please check the server setup.")

    def diconnect(self):
        try:
            self.client.disconnect()

        except RuntimeError:
            print("The real-time controller could not be stopped due to server issues. Please stop the client manually and delete the object!")
        
    def set_read_write(self, read='x', write='u'):
        self.readvariable = read
        self.writevariable = write

    def set_default_write_ns(self):
        if isinstance(self.def_namespace._namespace_index, int):
            for dclass in self.def_namespace.entry_list:
                if dclass.objectnode == self.writevariable:
                    self.tagout.append(dclass.get_node_id(self.def_namespace._namespace_index))
                elif dclass.objectnode == 'x':
                    self.init_server_tags.append(dclass.get_node_id(self.def_namespace._namespace_index))

            print('Default write-nodes set as {}.'.format(self.tagout))
        else:
            print('Namespace index is unknown. Please register this client on the target server first using: RTServer.namespace_from_client(Client)')


    def set_default_read_ns(self):
        if isinstance(self.def_namespace._namespace_index, int):
            for dclass in self.def_namespace.entry_list:
                if dclass.objectnode == self.readvariable:
                    self.tagin.append(dclass.get_node_id(self.def_namespace._namespace_index))

            print('Default read-nodes set as {}.'.format(self.tagin))

        else:
            print('Namespace index is unknown. Please register this client on the target server first using: RTServer.namespace_from_client(Client)')
    
    def return_namespace(self):
        return self.client.return_namespace()


    def readfrom(self, controller):
        self.client.register_namespace_from_client(controller)
        new_ns = self.return_namespace()[-1]

        if isinstance(new_ns._namespace_index, int):
            for dclass in new_ns.entry_list:
                if dclass.objectnode == self.readvariable:
                    self.tagin.append(dclass.get_node_id(new_ns._namespace_index))

            print('Read-nodes set as {}.'.format(self.tagin))

        else:
            print('Namespace index is unknown to target server. Please register this client on the target server first using: RTServer.namespace_from_client(Client)')

    def make_step(self):

        self.input = np.array([self.client.readData(i) for i in self.tagin]).reshape(-1,1)

        self.output = self.do_mpc_object.make_step(self.input)

        self.write_current(self.output)


    def write_current(self, data):        
        for count, item in enumerate(self.tagout):
            self.client.writeData([data.flatten()[count]], item)


    def async_run(self):
        self.is_running = False    
        self.async_step_start()
        self.make_step()


    def async_step_start(self):
        if self.new_init == True:
            self.new_thread = Thread(target=self.make_step)
            self.new_thread.start()
            self.new_init = False

        if not self.is_running:
            self.cycle = time.time() + self.cycle_time
            self.thread = Timer(self.cycle - time.time(), self.async_run)
            self.thread.start()
            self.is_running = True


    def async_step_stop(self ):
        self.thread.cancel()
        self.is_running = False
        self.new_init = True



#%%
@dataclass
class NamespaceEntry:
    objectnode: str
    variable: str
    dim: int

    def get_node_id(self, namespace_index):
        return f'ns={namespace_index};s={self.variable}'
    


@dataclass   
class Namespace:
    namespace_name: str
    entry_list: List[NamespaceEntry]
    _namespace_index: int = None

    def add_entries(self, entries: List[NamespaceEntry]):
        pass


    def has_entry(self, entry: NamespaceEntry):
        return (entry in self.entry_list)
    
    def return_ns_dict(self):
        return asdict(self)


    @property
    def namespace_index(self):
        if self._namespace_index == None:
            raise Exception('Namespace is not registered at server.')

        return self._namespace_index
    
    @namespace_index.setter
    def namespace_index(self, val):
        if not isinstance(val, int):
            raise ValueError('argument must be an integer')
        
        self._namespace_index = val

    #TODO: def __getitem__(self, index):
        # für besseres indexing
class namespace_from_model:    

    def summarized(model, model_name, object_name):
        node_list = []
        model_dict = {'aux':model.n_aux,
                        'p':model.n_p,
                            'tvp':model.n_tvp, 
                            'u':model.n_u, 
                            'v':model.n_v, 
                            'w':model.n_w, 
                            'x':model.n_x, 
                            'y':model.n_y, 
                            'z':model.n_z}
            
        for key in model_dict:
            node = NamespaceEntry(object_name, key, model_dict[key])
            node_list.append(node)
            
        return Namespace(model_name, node_list)

    def detailed(model, model_name='do_mpc_model'):
        node_list = []
        variable_list = ['aux', 'p', 'tvp', 'u', 'v', 'w', 'x', 'y', 'z']

        for var in variable_list:
            for key in model[var].keys():
                if key != 'default':
                    node_list.append(NamespaceEntry(var, key, 1))
                else:
                    continue

        return Namespace(model_name, node_list)

@dataclass
class ServerOpts:
    name: str
    address: str
    port: int
    db: bool

@dataclass
class ClientOpts:
    name: str
    address: str
    port: int



e1 = NamespaceEntry(
    'plant', 'state1.x', 2
)
e2 = NamespaceEntry(
    'pizza', 'state2.x', 2
)
e3 = NamespaceEntry(
    'plant', 'state3.x', 2
)

ns = Namespace('controller', [e1,e2])
ns2 = Namespace('estimator',[e3])
#%% Server setup


# Defining the settings for the OPCUA server
server_opts = ServerOpts("Bio Reactor OPCUA",      # give the server whatever name you prefer
               "opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               4840,                    # does not need changing
               False)                 # set to True if you plan to use the SQL database during/after the runtime 

server_opts_2 = ServerOpts("Bio Reactor OPCUA",      # give the server whatever name you prefer
               "opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               4841,                    # does not need changing
               False)   



client_opts_1 = ClientOpts("Bio Reactor OPCUA Client_1","opc.tcp://localhost:4840/freeopcua/server/",4840)
client_opts_2 = ClientOpts("Bio Reactor OPCUA Client_2","opc.tcp://localhost:4840/freeopcua/server/",4840)


#%%



import sys
from casadi import *

# Add do_mpc to path. This is not necessary if it was installed via pip
sys.path.append('../../../')

# Import do_mpc package:
import do_mpc

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

mpc.setup()

# Initial state
X_s_0 = 1.0 # Concentration biomass [mol/l]
S_s_0 = 0.5 # Concentration substrate [mol/l]
P_s_0 = 0.0 # Concentration product [mol/l]
V_s_0 = 120.0 # Volume inside tank [m^3]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])

mpc.x0 = x0
mpc.set_initial_guess()
# Client1 = RTClient(client_opts_1,ns)


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
simulator.x0 = x0
#%%
# Client1 = RTClient(client_opts_1,ns)
# Client2 = RTClient(client_opts_2,ns2)
Server = RTServer(server_opts)
# Server.namespace_from_client(Client1)
# Server.namespace_from_client(Client2)
rt_mpc = RTBase(mpc, client_opts_1, 'model1')
rt_sim = RTBase(simulator, client_opts_2, 'model2')
rt_mpc.set_read_write(read='x', write='u')
rt_sim.set_read_write(read='u', write='x')

Server.namespace_from_client(rt_mpc)
Server.namespace_from_client(rt_sim)
#%%
# Server.get_all_nodes()
rt_mpc.set_default_write_ns()
rt_sim.set_default_write_ns()
#%%
# rt_mpc.set_default_read_ns()
rt_mpc.readfrom(rt_sim)
rt_sim.readfrom(rt_mpc)
#%%
# Server.get_all_nodes()
Server.start()
rt_mpc.connect()
rt_sim.connect()

# rt_mpc.x0_server()
# rt_sim.x0_server()
# rt_mpc.make_step()
# rt_sim.init_server()
#%%
rt_mpc.async_step_start()
rt_sim.async_step_start()
#%%

#%%
# Server.stop()
# %%

for i in range(200):
    print({'u':rt_mpc.client.readData('ns=2;s=u[inp]'),
    'x':rt_mpc.client.readData('ns=3;s=x[X_s]')})
    time.sleep(5)

# %%
