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


import time
import numpy as np
import pandas as pd
import do_mpc
try:
    import asyncua.sync as opcua
except ImportError:
    raise ImportError("The asyncua library is not installed. Please install it and try again.")
        
from asyncua.server.history_sql import HistorySQLite


class RTServer:

    def __init__(self, opts):
        self.dtype = 'default'
        model = opts['_model']
       
        # The basic OPCUA server definition contains a name, address and a port numer
        # The user can decide if they want to activate the SQL database option (_with_db=TRUE)
        self.name    = opts['_name']
        self.address = opts['_address']
        self.port    = opts['_port']
        self.with_db = opts['_with_db']
        

        # Dictionary with possible data_fields in the class and their respective dimension. All data is numpy ndarray.
        self.data_structure = {
            'nr_x_states': model.n_x,
            'nr_z_states': model.n_z, 
            'nr_inputs'  : model.n_u,
            'nr_meas'    : model.n_y,
            'nr_controls': model.n_u,
            'nr_tv_pars' : model.n_tvp,
            'nr_mod_pars': model.n_p,
            'nr_aux'     : model.n_aux
        }
        
        """ 
        The user defined server namespace to be implemented on the OPCUA server. Contains pairs of the form (elementary MPC variable - readable user name) 
        """
        self.namespace = {
            'PlantData'      :{'x':"States.X",'z':"States.Z",'u':"Inputs",'y':"Measurements",'p':"Parameters"},
            'ControllerData' :{'x_init':"InitialState",'u_opt':"OptimalOutputs",'x_pred': "PredictedStates", 'u_pred': "PredictedOutputs"},
            'EstimatorData'  :{'xhat':"Estimates.X",'zhat':"Estimates.Z",'phat':"Estimates.P"}
            }
        try:
            self.opcua_server = opcua.Server()
        except RuntimeError:
            self.created = False
            print("Server could not be created. Check your opcua module installation!")
            return False
        
        self.opcua_server.set_endpoint(self.address)
        self.opcua_server.set_server_name(self.name)
        
        # Setup a default namespace, because personalizing it does not bring any value for now
        idx = self.opcua_server.register_namespace("Realtime NMPC structure")


        objects = self.opcua_server.nodes.objects.add_object(idx, "PlantData")
        

        placeholder = [0 for x in range(self.data_structure['nr_x_states'])]
        datavector = objects.add_variable(opcua.ua.NodeId("States.X", idx), "States.X", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_z_states'])] if self.data_structure['nr_z_states']>0 else [0]
        datavector = objects.add_variable(opcua.ua.NodeId("States.Z", idx), "States.Z",  placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_meas'])] if self.data_structure['nr_meas']>0 else [0]
        datavector = objects.add_variable(opcua.ua.NodeId("Measurements", idx), "Measurements", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_inputs'])] if self.data_structure['nr_inputs']>0 else [0]
        datavector = objects.add_variable(opcua.ua.NodeId("Inputs", idx), "Inputs", placeholder)
        datavector.set_writable()

        objects = self.opcua_server.nodes.objects.add_object(idx, "ControllerData")
        
        placeholder = [0 for x in range(self.data_structure['nr_x_states'])] if self.data_structure['nr_x_states']>0 else [0]
        datavector = objects.add_variable(opcua.ua.NodeId("InitialState", idx), "InitialState", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_inputs'])] if self.data_structure['nr_inputs']>0 else [0]
        datavector = objects.add_variable(opcua.ua.NodeId("OptimalOutputs", idx), "OptimalOutputs", placeholder)
        datavector.set_writable()
 
        
        objects = self.opcua_server.nodes.objects.add_object(idx, "EstimatorData")
        
        placeholder = [0 for x in range(self.data_structure['nr_x_states'])]
        datavector = objects.add_variable(opcua.ua.NodeId("Estimates.X", idx), "Estimates.X",  placeholder)
        datavector.set_writable()
        if self.data_structure['nr_mod_pars'] > 0:
            placeholder = [0 for x in range(self.data_structure['nr_mod_pars'])]
            datavector = objects.add_variable(opcua.ua.NodeId("Estimates.P", idx), "Estimates.P",  placeholder)
            datavector.set_writable()
            
        
        # Mark the server as created & not yet running
        self.created = True
        self.running = False
    
        # # Initiate SQL database
        # if self.with_db == True:
        #     self.opcua_server.history_manager.set_storage(HistorySQLite('node_history.sql'))
            
    # # Get a list with all node ID's for the SQL database    
    # def get_all_nodes(self):        
    #     node_list = []
    #     for ns_id in self.namespace.keys():
    #         for node_id in self.namespace[ns_id].items():
    #             node_list.append(self.opcua_server.get_node('ns=2;s=' + node_id[1]))
    #     return node_list
            
    # Start server
    def start(self):
        try:
            self.opcua_server.start()
            if self.with_db == True:
                for it in self.get_all_nodes():
                    try:
                        self.opcua_server.historize_node_data_change(it,count=1e6)
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
        
    
    # def extract_data(self, node, data_structure):
    #     data = self.opcua_server.get_node(node).read_raw_history()
    #     val = np.array([np.asarray(val.Value.Value).flatten() for val in data]).reshape(-1, data_structure)
    #     timestamp = np.array([val.SourceTimestamp for val in data]).reshape(-1,1)
    #     labels = ['{}_{}'.format(node.split(";")[-1], ind) for ind in range(len(val[1]))]
    #     df = pd.DataFrame(val, columns=labels)
    #     df['Timestamp'] = timestamp
    #     return df
    
    # def get_data(self):
    #     df_states = self.extract_data('ns=2;s=Measurements', self.data_structure['nr_meas'])
    #     df_outputs = self.extract_data('ns=2;s=OptimalOutputs', self.data_structure['nr_controls'])
    #     return df_states, df_outputs
    
    
#%%

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
               "_name":"Bio Reactor OPCUA",      # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4840,                    # does not need changing
               "_with_db": False}                 # set to True if you plan to use the SQL database during/after the runtime 
# Create your OPUA server