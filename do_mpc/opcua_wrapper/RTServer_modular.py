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
            if key_0._namespace_index == None:
                idx = self.opcua_server.register_namespace(key_0.namespace_name)
            else:
                print(f"The namespace {key_0.namespace_name} was already registered on this server.")
                continue
            

            for key_1 in key_0.entry_list:
                # TODO: if condition such that every object is only registered once
                try:
                    objects = self.opcua_server.nodes.objects.add_object(idx, key_1.objectnode)
                    placeholder = [0] * key_1.dim
                    datavector = objects.add_variable(opcua.ua.NodeId(key_1.variable, idx), key_1.variable, placeholder)
                    datavector.set_writable()

                except:
                    placeholder = [0] * key_1.dim
                    datavector = objects.add_variable(opcua.ua.NodeId(key_1.variable, idx), key_1.variable, placeholder)
                    datavector.set_writable()


                datavector.set_writable()

            client.add_namespace_url(key_0, idx, self.name)
            
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
        self.namespace_list += client.return_namespace()
    
    def add_namespace_url(self, key, url, server_name):
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
    

# class RealtimeController:
#     def __init__(self, controller, client_opts):
#         self.write_nodes = []
#         self.read_nodes = []

#         read_node_list = generate_write_nodes_list()

#         self.client = Client(client_opts, read_node_list)

#     def generate_write_nodes_list(self):
#         entry_list = []

#         for key in self.controller.model.u.keys.keys():
#             entry_list.append(
#                 NamespaceEntry(
#                     objectnode ='controller',
#                     variable = key,
#                     dim = 3 #TODO fix this
#                 )
#             )

#         return entry_list
    
#     def read_from_(object):
#         #TODO: fix this

# class RTController:

#     def __init__(self, Controller, ClientOpts):
#         self.mpc = Controller
#         self.client = RTClient(ClientOpts)

#         self.mpc.model

#     def init_server(self):

#         tag = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
#         dataVal = np.array(vertcat(self.u0)).flatten().tolist()
#         try:
#             self.opc_client.writeData(dataVal, tag)
#         except RuntimeError:
#             self.enabled = False
#             print("The real-time controller could not connect to the server. Please correct the server setup.")
        
#         # One optimizer iteration is called before starting the cyclical operation
        
#         self.asynchronous_step()
        
#         return self.enabled
    
#     def start(self):
#         """Alternative method to start the client from the console. The client is usually automatically connected upon instantiation.
        
#         :return result: The result of the connection attempt to the OPC-UA server.
#         :rtype result: boolean
#         """
#         try:
#             self.opc_client.connect()
#             self.enabled = True
#         except RuntimeError:
#             self.enabled = False
#             print("The real-time controller could not connect to the server. Please check the server setup.")
#         return self.enabled
        
#     def stop(self):
#         """ Stops the execution of the real-time estimator by disconnecting the OPC-UA client from the server. 
#         Throws an error if the operation cannot be performed.
        
#         :return result: The result of the disconnect operation
#         :rtype: boolean
#         """
#         try:
#             self.opc_client.disconnect()
#             self.enabled = False
#         except RuntimeError:
#             print("The real-time controller could not be stopped due to server issues. Please stop the client manually and delete the object!")
#         return self.enabled
    
#     def check_status(self):
#         """This function is called before every optimization step to ensure that the server data
#         is sane and a call to the optimizer can be made in good faith, i.e. the plant state
#         contains meaningful data and that no flags have been raised.
        
#         :param no params: this function onyl needs internal data
        
#         :return: check_result is the result of all the check done by the controller before executing the step
#         :rtype: boolean
#         """
#         check_result = self.is_ready
#         # Step 1: check that the server is running and the client is connected
#         check_result = self.opc_client.connected and check_result
#         if check_result == False: 
#             print("The controller check failed because: controller not connected to server.")
#             return False
        
#         # Step 2: check whether the user has requested to run the optimizer
#         if self.user_controlled:
#             check_result = check_result and self.opc_client.checkSwitches(pos=0)
#             if check_result == False: 
#                 print("The controller check failed because: controller not manually enabled on the server.")
#                 return False
        
#         # Step 3: check whether the controller should run and no controller flags have been raised
#         # flags = [0-controller, 1-simulator, 2-estimator, 3-monitoring, 4-extra]
#         check_result = check_result and not self.opc_client.checkFlags(pos=0)
#         if check_result == False: 
#             print("The controller check failed because: controller has raised a failure flag.")
#             return False
        
#         # Step 4: check that the plant/simulator is running and no simulator flags have been raised
#         check_result = check_result and  not (self.opc_client.checkFlags(pos=1) or self.opc_client.checkFlags(pos=2))
#         self.is_ready = check_result
#         if check_result == False: 
#             print("The controller check failed because: either the simulator or estimator have reported crap data. Unsafe to run the controller!")
#         return check_result
    

        
#     def asynchronous_step(self):
#         """This function implements the server calls and simulator step with a predefined frequency
#         :param no params: because the cycle is stored by the object
        
#         :return: time_left, the remaining time on the clock when the optimizer has finished the routine
#         :rtype: float
#         """
#         if self.output_feedback == False:
#             tag_in  = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
#         else:
#             tag_in  = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        
#         tag_out = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
#         # tag_out_pred = "ns=2;s="+self.opc_client.namespace['ControllerData']['x_pred']
#         # Read the latest plant state from server and execute optimization step
#         xk = np.array(self.opc_client.readData(tag_in))
        
#         # The NLP must be reinitialized with the most current data from the plant readings
#         self.x0 = np.array(self.opc_client.readData(tag_in))   
#         self.set_initial_guess()

#         # Check the current status before running the optimizer step 
#         if self.check_status():
#             # The controller can be executed
#             uk = self.make_step(xk)
#             x_pred = self.opt_x_num_unscaled
#             # The iteration count is incremented regardless of the outcome
#             self.iter_count = self.iter_count + 1
            
#             if self.solver_stats['return_status'] == 'Solve_Succeeded':
#                 # The optimal inputs are written back to the server
#                 self.opc_client.writeData(uk.tolist(), tag_out)
#                 # self.opc_client.writeData(np.array(vertcat(x_pred)).tolist(), tag_out_pred)
#             else:
#                 print("The controller failed at time ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
#                 print("The optimal inputs have not been updated on the server.")
#             # The controller must wait for a predefined time
#             time_left = self.cycle_time-self.solver_stats['t_wall_total']
            
#         else: 
#             time_left = self.cycle_time
#             print("The controller is still waiting to be manually activated. When you're ready set the status bit to 1.")
        
#         return time_left
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
    

def namespace_from_model(model, model_name, object_name):
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
Client1 = RTClient(client_opts_1,ns)
Client2 = RTClient(client_opts_2,ns2)
Server = RTServer(server_opts)
Server.namespace_from_client(Client1)
Server.namespace_from_client(Client2)

#%%
# Server.get_all_nodes()
Server.start()

#%%
Server.stop()
# %%
