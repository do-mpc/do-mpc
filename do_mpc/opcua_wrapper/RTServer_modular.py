
#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
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
from threading import Timer, Thread
from casadi import *

try:
    import asyncua.sync as opcua
except ImportError:
    raise ImportError("The asyncua library is not installed. Please install it and try again.")
        


class RTServer:

    def __init__(self, opts):
       
        # The basic OPCUA server definition contains a name, address and a port numer
        self.name    = opts.name
        self.address = opts.address
        self.port    = opts.port

        # Try to create the server
        try:
            self.opcua_server = opcua.Server()
            print("A server named - {} - was created".format(self.name))
        except RuntimeError:
            self.created = False
            print("Server could not be created. Check your opcua module installation!")
            return False

        
    def namespace_from_client(self, client):
        # Obtain client namespace
        client_namespace = client.client.return_namespace()
        self.object_dict = {}  
        # Create the obtained namespace on the server, if it doesent already exist
        if client_namespace._namespace_index == None:
            idx = self.opcua_server.register_namespace(client_namespace.namespace_name)
        else:
            print(f"The namespace {client_namespace.namespace_name} was already registered on this server.")
        # Populate namespace with object nodes and variables
        # Iterate through all namespace entries
        for namespace_entry in client_namespace.entry_list: 
            # Check if the variable even exists inside the do-mpc model
            if  namespace_entry.dim > 0: 
                # Check if the object node specified inside the namespace_entry dataclass was alreade created
                if namespace_entry.objectnode not in self.object_dict.keys(): 
                    # Create the object node
                    object = self.opcua_server.nodes.objects.add_object(idx, namespace_entry.objectnode)
                    # Add object node to object_node dict, used to make sure no object node is created twice
                    self.object_dict[namespace_entry.objectnode] = object
                    # Add variable to object node
                    self.add_variable_to_node(namespace_entry, idx) 

                else:
                    self.add_variable_to_node(namespace_entry, idx)

            # Pass iteration if variable doesent exists
            else:
                continue
        
        # Write namespace index into client namespace so the client knows where to find the variables
        client.client.add_namespace_url(idx)     
        print(f"The following namespaces are registered: {self.opcua_server.get_namespace_array()[2:]}")


    def add_variable_to_node(self, NamespaceEntry, Namespace_url):
        # Create a placholder list containing floats for each variable
        placeholder = [0.0] * NamespaceEntry.dim
        # Create descriptive variable names: Object_node[Variavle]
        variable_name = NamespaceEntry.objectnode + '[' + NamespaceEntry.variable + ']'
        # Add variable to object node
        datavector = self.object_dict[NamespaceEntry.objectnode].add_variable(opcua.ua.NodeId(variable_name, Namespace_url), variable_name, placeholder)
        datavector.set_writable()
        # Write descreptive wariable name to client namespace
        NamespaceEntry.variable = variable_name

    
    # Start server
    def start(self):
        try:
            self.opcua_server.start()

        except RuntimeError as err:
            print("The server "+ self.name +" could not be started, returned error message :\n", err)

    
        
    # Stop server
    def stop(self):
        try:
            self.opcua_server.stop()
            print("The server  "+ self.name +" was stopped successfully @ ",time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))

        except RuntimeError as err:
            print("The server could not be stopped, returned error message :\n", err)



class RTClient:

    def __init__(self, opts, namespace):       
        # Information for connection to server
        self.server_address = opts.address
        self.port           = opts.port
        self.name           = opts.name
        
        # Information relevant for the client specific namespace
        self.namespace = namespace

        # Create Client and check if server is responding
        try:
            self.opcua_client = opcua.Client(self.server_address)
            print("A client named -", self.name, "- was created")

        except RuntimeError:
            print("The connection to the server could not be established\n", self.server_address, "is not responding")


    def return_namespace(self):        
            return self.namespace
    

    # Method used by server to mark namespace with the corresponding url
    def add_namespace_url(self, url):
        self.namespace._namespace_index = url

    # This function implements (re)connection to the designated server
    def connect(self):
        try:
            self.opcua_client.connect()
            print("The -", self.name, "- has just connected to ",self.server_address)

        except RuntimeError:
            print("The connection to the server could not be established\n", self.server_address,
                  " is not responding")


    def disconnect(self):
        self.opcua_client.disconnect()
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
    

class RTBase:

    def __init__(self, do_mpc_object, clientOpts, namespace=None):
        self.do_mpc_object = do_mpc_object

        if namespace == None:
            self.get_default_namespace(clientOpts.name)
        else:
            self.def_namespace = namespace

        self.cycle_time = do_mpc_object.t_step
        self.client = RTClient(clientOpts, self.def_namespace)
        self.tagout = []
        self.tagin = []
        self.is_running = False
        self.new_init = True


    def get_default_namespace(self, namespace_name):
        self.def_namespace = namespace_from_model(self.do_mpc_object.model, namespace_name)


    def connect(self):
        try:
            self.client.connect()
        except RuntimeError:
            self.enabled = False


    def disconnect(self):
        try:
            self.client.disconnect()
        except RuntimeError:
            print("The real-time controller could not be stopped due to server issues. Please stop the client manually and delete the object!")


    def set_write_tags(self,tagout:List[str]):
        self.tagout = tagout


    def set_read_tags(self, tagin:List[str]):
        self.tagin = tagin


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


    def async_step_stop(self):
        self.thread.cancel()
        self.is_running = False
        self.new_init = True

    def init_server(self):
        self.write_current(np.array(vertcat(self.do_mpc_object.x0)))


@dataclass
class NamespaceEntry:
    objectnode: str
    variable: str
    dim: int

    def get_node_id(self, namespace_index):
        if namespace_index == None:
            raise Exception('Namespace_index not defined')
        return f'ns={namespace_index};s={self.variable}'
    


@dataclass   
class Namespace:
    namespace_name: str
    entry_list: List[NamespaceEntry]
    _namespace_index: int = None

    def __getitem__(self, nodename: str):
        return [entry.get_node_id(self._namespace_index) for entry in self.entry_list if entry.objectnode == nodename]


def namespace_from_model(model, model_name):
    node_list = []
    variable_list = ['aux', 'p', 'tvp', 'u', 'v', 'w', 'x', 'y', 'z']

    for var in variable_list:
        for key in model[var].labels():
            key = key.strip('[]').split(',')
            if key[0] != 'default':
                node_list.append(NamespaceEntry(var, key[0], 1 + int(key[-1])))
            else:
                continue

    return Namespace(model_name, node_list)


@dataclass
class ServerOpts:
    name: str
    address: str
    port: int


@dataclass
class ClientOpts:
    name: str
    address: str
    port: int


