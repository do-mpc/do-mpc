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
try:
    import opcua
except ImportError:
    raise ImportError("The opcua library is not installed. Please install it and try again.")
        
from opcua.server.history_sql import HistorySQLite


class Server:
    """**do-mpc** OPCUA Server. An instance of this class is created for all active **do-mpc** classes,
    e.g. :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.estimator.EKF`.

    The class is initialized with relevant instances of the **do-mpc** configuration, e.g :py:class:`do_mpc.model.Model`, :py:class:`do_mpc.model.Optimizer` etc, which contain all
    information about variables (e.g. states, inputs, optimization outputs etc.).

    The :py:class:`Server` class has a public API, which can be used to manually create and launch a server from the interpreter. However, it is recommended to use the real-time manager object
    :py:class:`Manager`
    """
    def __init__(self, opts):
        self.dtype = 'default'
        model = opts['_model']
       
        # The basic OPCUA server definition contains a name, address and a port numer
        # The user can decide if they want to activate the SQL database option (_with_db=TRUE)
        self.name    = opts['_name']
        self.address = opts['_address']
        self.port    = opts['_port']
        self.with_db = opts['_with_db']
        
        # The server type describes one of three levels of complexity: 
        # _server_type = 'basic' --> only necessary data structures are setup (functioning optimizer)
        # _server_type = 'with_estimator'  --> if an MHE or EKF are implemented, the estimated states are stored
        # _server_Type = 'with_monitoring'  --> the server also stores additional KPIs for monitoring
        self.server_type = opts['_server_type']
        if self.server_type == 'basic': print ("Server setup #1: You have opted for the basic data server!")
        elif self.server_type == 'with_estimator': print ("Server setup #1: You have opted for the data server with state and parameter estimates!")
        else: print("Server setup #1: You have opted for the full available data structure!")
        
        # If True, all parameters will be stored on the server
        self.store_params = opts['_store_params']
        if self.store_params: print("Server setup #2: The model parameters will be stored on the server.")
        else: print("Server setup #2: Parameters will not be stored on the server!") 
        
        # If True, the predictions of the optimizer are also stored 
        self.store_predictions = opts['_store_predictions']
        if self.store_predictions : 
            print("Server setup #3: The OPCUA server will have the predictions available at runtime.")
            n_steps_pred = opts['_n_steps_pred']
        else: 
            print ("Server setup #3: The OPCUA server will not have the predictions available at runtime.")
            n_steps_pred = 0

        # Dictionary with possible data_fields in the class and their respective dimension. All data is numpy ndarray.
        self.data_structure = {
            'nr_x_states': model.n_x,
            'nr_z_states': model.n_z, #(add model.n_aux)
            'nr_inputs'  : model.n_u,
            'nr_meas'    : model.n_y,
            'nr_controls': model.n_u,
            'nr_tv_pars' : model.n_tvp,
            'nr_mod_pars': model.n_p,
            'nr_aux'     : model.n_aux,
            'nr_x_pred'  : model.n_x * n_steps_pred ,
            'nr_u_pred'  : model.n_u * n_steps_pred ,
            'nr_flags'   : 5,
            'nr_switches': 5
        }
        """ 
        The user defined server namespace to be implemented on the OPCUA server. Contains pairs of the form (elementary MPC variable - readable user name) 
        """
        self.namespace = {
            'PlantData'      :{'x':"States.X",'z':"States.Z",'u':"Inputs",'y':"Measurements",'p':"Parameters"},
            'ControllerData' :{'x_init':"InitialState",'u_opt':"OptimalOutputs",'x_pred': "PredictedStates", 'u_pred': "PredictedOutputs"},
            'EstimatorData'  :{'xhat':"Estimates.X",'zhat':"Estimates.Z",'phat':"Estimates.P"},
            'SupervisionData':{'flags':"Flags",'switches':"Switches"}
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

        # Get objects node, this is where the nodes are put
        objects = self.opcua_server.get_objects_node()
        
        # Create the basic data structure, which consists of the simulator and the optimizer
        localvar = objects.add_object(opcua.ua.NodeId("PlantData", idx), "PlantData")
        
        placeholder = [0 for x in range(self.data_structure['nr_x_states'])]
        datavector = localvar.add_variable(opcua.ua.NodeId("States.X", idx), "States.X", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_z_states'])] if self.data_structure['nr_z_states']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("States.Z", idx), "States.Z",  placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_meas'])] if self.data_structure['nr_meas']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Measurements", idx), "Measurements", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_inputs'])] if self.data_structure['nr_inputs']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Inputs", idx), "Inputs", placeholder)
        datavector.set_writable()
        if self.store_params == True:
            placeholder = [0 for x in range(self.data_structure['nr_mod_pars'])] if self.data_structure['nr_mod_pars']>0 else [0]
            datavector = localvar.add_variable(opcua.ua.NodeId("Parameters", idx), "Parameters", placeholder)
            datavector.set_writable()
        
        localvar = objects.add_object(opcua.ua.NodeId("ControllerData", idx), "ControllerData")
        
        placeholder = [0 for x in range(self.data_structure['nr_x_states'])] if self.data_structure['nr_x_states']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("InitialState", idx), "InitialState", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_inputs'])] if self.data_structure['nr_inputs']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("OptimalOutputs", idx), "OptimalOutputs", placeholder)
        datavector.set_writable()
        if self.store_params == True:
            placeholder = [0 for x in range(self.data_structure['nr_u_pred'])] if self.data_structure['nr_u_pred']>0 else [0]
            datavector = localvar.add_variable(opcua.ua.NodeId("PredictedOutputs", idx), "PredictedOutputs",  placeholder)
            datavector.set_writable()
        if self.store_predictions == True:
            placeholder = [0 for x in range(self.data_structure['nr_x_pred'])] if self.data_structure['nr_x_pred']>0 else [0]
            datavector = localvar.add_variable(opcua.ua.NodeId("PredictedStates", idx), "PredictedStates", placeholder)
            datavector.set_writable()    
        
        if self.server_type == 'with_estimator':
            localvar = objects.add_object(opcua.ua.NodeId("EstimatorData", idx), "EstimatorData")
            placeholder = [0 for x in range(self.data_structure['nr_x_states'])]
            datavector = localvar.add_variable(opcua.ua.NodeId("Estimates.X", idx), "Estimates.X",  placeholder)
            datavector.set_writable()
            if self.data_structure['nr_mod_pars'] > 0:
                placeholder = [0 for x in range(self.data_structure['nr_mod_pars'])]
                datavector = localvar.add_variable(opcua.ua.NodeId("Estimates.P", idx), "Estimates.P",  placeholder)
                datavector.set_writable()
            
        # The flags are defined by default 
        localvar = objects.add_object(opcua.ua.NodeId("UserData", idx), "UserData")
        
        placeholder = [0 for x in range(self.data_structure['nr_flags'])] if self.data_structure['nr_flags']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Flags", idx), "Flags",  placeholder)
        datavector.set_writable()
        # The switches allow for manual control of the real-time modules remotely
        placeholder = [0 for x in range(self.data_structure['nr_switches'])] if self.data_structure['nr_switches']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Switches", idx), "Switches", placeholder)
        datavector.set_writable()
        
        # Mark the server as created & not yet running
        self.created = True
        self.running = False
    
        # Initiate SQL database
        if self.with_db == True:
            self.opcua_server.iserver.history_manager.set_storage(HistorySQLite('node_history.sql'))
            
    # Get a list with all node ID's for the SQL database    
    def get_all_nodes(self):        
        node_list = []
        for ns_id in self.namespace.keys():
            for node_id in self.namespace[ns_id].items():
                node_list.append(self.opcua_server.get_node('ns=2;s=' + node_id[1]))
        return node_list
            
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
        
    
    def extract_data(self, node, data_structure):
        data = self.opcua_server.get_node(node).read_raw_history()
        val = np.array([np.asarray(val.Value.Value).flatten() for val in data]).reshape(-1, data_structure)
        timestamp = np.array([val.SourceTimestamp for val in data]).reshape(-1,1)
        labels = ['{}_{}'.format(node.split(";")[-1], ind) for ind in range(len(val[1]))]
        df = pd.DataFrame(val, columns=labels)
        df['Timestamp'] = timestamp
        return df
    
    def get_data(self):
        df_states = self.extract_data('ns=2;s=Measurements', self.data_structure['nr_meas'])
        df_outputs = self.extract_data('ns=2;s=OptimalOutputs', self.data_structure['nr_controls'])
        return df_states, df_outputs
    
    
