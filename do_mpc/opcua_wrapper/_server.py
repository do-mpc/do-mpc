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
        client_namespace = client.client.namespace
        self.object_node_dict = {}
        idx = self.opcua_server.register_namespace(client_namespace.namespace_name)

        for namespace_entry in client_namespace.entry_list: 
            if namespace_entry.objectnode not in self.object_node_dict.keys():
                object = self.opcua_server.nodes.objects.add_object(idx, namespace_entry.objectnode)
                self.object_node_dict[namespace_entry.objectnode] = object

            self.add_variable_to_node(namespace_entry, idx)

        client.client.add_namespace_url(idx)     
        print(f"The following namespaces are registered: {self.opcua_server.get_namespace_array()[2:]}")


    def add_variable_to_node(self, NamespaceEntry, Namespace_url):
        variable_name = f'{NamespaceEntry.objectnode}{NamespaceEntry.variable}'
        datavector = self.object_node_dict[NamespaceEntry.objectnode].add_variable(opcua.ua.NodeId(variable_name, Namespace_url), variable_name, [0.0])
        datavector.set_writable()
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

