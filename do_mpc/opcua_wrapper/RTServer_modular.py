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


import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

import do_mpc
try:
    import asyncua.sync as opcua
except ImportError:
    raise ImportError("The asyncua library is not installed. Please install it and try again.")
        
from asyncua.server.history_sql import HistorySQLite


class RTServer:

    def __init__(self, opts):
       
        # The basic OPCUA server definition contains a name, address and a port numer
        # The user can decide if they want to activate the SQL database option (_with_db=TRUE)
        self.name    = opts['_name']
        self.address = opts['_address']
        self.port    = opts['_port']
        self.with_db = opts['_with_db']
        

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
        self.namespace_dict = client.return_namespace()
        
        # Create the obtained namespaces on the server, if they do not already exist
        for key_0 in self.namespace_dict:
            ns_flag = self.name in self.namespace_dict[key_0]
            if not ns_flag:
                idx = self.opcua_server.register_namespace(key_0)
            else:
                print(f"The namespace {key_0} was already registered on this server.")
                continue
            
            # Set up object structure using another loop for the sub-dict structure (objects representing plant, controller, etc.)
            for key_1 in self.namespace_dict[key_0]:
                if isinstance(self.namespace_dict[key_0][key_1], dict):
                    objects = self.opcua_server.nodes.objects.add_object(idx, key_1)
                else:
                    continue
                    
                # Add variables for all objects
                for key_2 in self.namespace_dict[key_0][key_1]:
                    sub_value = self.namespace_dict[key_0][key_1][key_2]
                    if isinstance(sub_value, int):
                        placeholder = [0] * sub_value
                        datavector = objects.add_variable(opcua.ua.NodeId(key_2, idx), key_2, placeholder)
                        datavector.set_writable()
                    else:
                        print(f"Key error: {key_2} does not exist in sub-dictionary. Please check the structure of your namespace dict: <NS_name: <Object_name: <Variavle_name: Variable_dimension>>>")
                        
            client.add_namespace_url(key_0, idx, self.name)
            
        self.opcua_server.get_namespace_array()
        print(f"The following namespaces are registered: {self.opcua_server.get_namespace_array()[2:]}")
    

    # Get a list with all node ID's for the SQL database    
    def get_all_nodes(self):        
        node_list = []
        for ns_id in self.namespace_dict:            
            for object_id in self.namespace_dict[ns_id]:
                if type(self.namespace_dict[ns_id][object_id]) == dict:
                    for node_id in  self.namespace_dict[ns_id][object_id]:
                            node_list.append(self.opcua_server.get_node('ns={};s='.format(self.namespace_dict[ns_id][self.name]) + node_id))
                else:
                    continue
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
        self.server_address = opts['_address']
        self.port           = opts['_port']
        self.name           = opts['_name']
        
        # Information relevant for the client specific namespace
        self.namespace_dict = {}
        self.namespace_dict = write_namespace

        # Create Client and check if server is responding
        try:
            self.opcua_client = opcua.Client(self.server_address)
            print("A client named -", self.name, "- was created")
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The connection to the server could not be established\n", self.server_address, "is not responding")

    def return_namespace(self):        
        # Returns all registered namespaces
            return self.namespace_dict
    
    def register_namespace_from_client(self, client):
        # Register namspaces from another client
        client_namespace_dict = client.return_namespace()     
        # Write new namespaces to namespace dict
        for key in client_namespace_dict:
            self.namespace_dict[key] = client_namespace_dict[key]
    
    def add_namespace_url(self, key, url, server_name):
        # Method used by server to mark namespaces with the corresponding url
        self.namespace_dict[key][server_name] = url




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
        """ Writes a tag onto the OPCUA server. It is used to write MPC data (states, parameters and inputs) and is called by methods
            from any of the real-time MPC modules. Asserts if the string has the right format and whether the data is a list.
            
            :param dataVal: a list of data to be wtitten on a tag
            :type dataVal: list
            
            :param tag: a name representing a valid tag on the server to which the client is connected.
            :type tag: string
            
            :return wr_result: The writing result as returned by the OPCUA writing method
            :rtype wr_result: boolean
        """
        assert type(dataVal) == list, "The data you provided is not arranged as a list. See the instructions for passing data to the server."
        assert "ns=" in tag, "The data destination you have provided is invalid. Refer to the OPCUA server namespace and define a correct source."
        
        try:
            wr_result = self.opcua_client.get_node(tag).set_value(dataVal)
        except ConnectionRefusedError:
            print("Write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return False
        
        return wr_result

    def readData(self, tag: NamespaceEntry):
        """ Reads values from a tag on the OPCUA server. It is used to read MPC data (states, parameters and inputs) and is called by methods
            from any of the real-time MPC modules. Asserts if the string has the right format and throws an error is the read operation has failed.
            
            :param tag: a name representing a valid tag on the server to which the client is connected.
            :type tag: string
            
            :return dataVal: The data list read from the server
            :rtype dataVal: list
            """
        
        if not self.namespace.has_entry(tag):
            raise Exception('blabla')
        
        query = tag.get_node_id(self.namespace.namespace.index)
           
        try:
            dataVal = self.opcua_client.get_node(query).get_value()
        except ConnectionRefusedError:
            print("A read operation by:", self.type, "failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        return dataVal
    

class RealtimeController:
    def __init__(self, controller, client_opts):
        self.write_nodes = []
        self.read_nodes = []

        read_node_list = generate_write_nodes_list()

        self.client = Client(client_opts, read_node_list)

    def generate_write_nodes_list(self):
        entry_list = []

        for key in self.controller.model.u.keys.keys():
            entry_list.append(
                NamespaceEntry(
                    objectnode ='controller',
                    variable = key,
                    dim = 3 #TODO fix this
                )
            )

        return entry_list
    
    def read_from_(object):
        #TODO: fix this
        

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
    

@dataclass
class ServerOpts:
    name: str

@dataclass 
class ClientOpts:
    name: str


e1 = NamespaceEntry(
    'plant', 'state.x', 2
)
e2 = NamespaceEntry(
    'plant', 'state.x', 2
)

ns = Namespace('controller', [e1,])

#%% Server setup


# Defining the settings for the OPCUA server
server_opts = {"_name":"Bio Reactor OPCUA",      # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4840,                    # does not need changing
               "_with_db": False}                 # set to True if you plan to use the SQL database during/after the runtime 

server_opts_2 = {"_name":"Bio Reactor OPCUA 2",      # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4841,                    # does not need changing
               "_with_db": False}   

test_ns_1 = {
    'Bio_stuff_1':{
        'Plant':{'State.X':3},
        'Controller':{'Input.u':2}
        }
    }

test_ns_2 = {
    'Bio_stuff_2':{
        'Plant':{'State.X':4, 'Measurements':3},
        'Controller':{'Input.u':3}
        }
    }

test_ns_3 = {
    'Bio_stuff_3':{
        'Plant':{'State.X':4, 'Measurements':3},
        'Controller':{'Input.u':3, 'Pizzas to order':4}
        }
    }



client_opts_1 = {"_name":"Bio Reactor OPCUA Client_1",      # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4840}   

client_opts_2 = {"_name":"Bio Reactor OPCUA Client_2",      # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4840} 

client_opts_3 = {"_name":"Bio Reactor OPCUA Client_3",      # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4840}   

client_opts_4 = {"_name":"Bio Reactor OPCUA Client_4",      # give the server whatever name you prefer
               "_address":"opc.tcp://localhost:4840/freeopcua/server/",  # does not need changing
               "_port": 4840}   

#%%

Server = RTServer(server_opts)
Server_2 = RTServer(server_opts_2)
Client_1 = RTClient(client_opts_1,test_ns_1)
Client_2 = RTClient(client_opts_2,test_ns_2)
Client_3 = RTClient(client_opts_3,test_ns_3)

#%%

Client_2.register_namespace_from_client(Client_3)
Client_2.return_namespace()
Client_1.register_namespace_from_client(Client_2)
#%%
Server.namespace_from_client(Client_1)
Server.namespace_from_client(Client_1)
#%%
# Server.get_all_nodes()
Server.start()

#%%
Server.stop()
# %%
