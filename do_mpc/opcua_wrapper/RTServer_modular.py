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
from casadi import *
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
        self.namespace += client.client.return_namespace()
        
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
        self.namespace_list.append(client.client.return_namespace()[0])
    
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


    def readfrom(self, controller):
        self.client.register_namespace_from_client(controller)
        new_ns = self.client.return_namespace()[-1]
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


