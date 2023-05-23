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
from ._base import RTBase
from ._helper import NamespaceEntry, ServerOpts
from ._client import RTClient


try:
    import asyncua.sync as opcua
except ImportError:
    raise ImportError("The asyncua library is not installed. Please install it and try again.")



class RTServer:
    '''
    Real Time Server.
    The RTServer class extends do-mpc with an easy to setup opcua server.

    **Configuration and setup:**

    Configuring and setting up the RTServer client involves the following steps:

    1. Use :py:class:`do_mpc.opcua.ServerOpts` dataclass to specify server name as well as IP adress and port for the server.

    2. Initiate the RTServer class with the ServerOpts dataclass.

    3. Use the :py:meth:`namespace_from_client` to automatically generate a namespace from a :py:class:`do_mpc.opcua.RTBase` instance (optional).

    4. Start the OPC UA server by calling :py:meth:`start`

    Note:

        Remember to properly stop the server afterwards using the :py:meth:`stop` method.
    
    Args:
        opts : Server options.
    
    '''

    def __init__(self, opts:ServerOpts)->None:
       
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

     
    def namespace_from_client(self, client:RTClient)->None:
        '''
        Takes an instance of :py:class:`do_mpc.opcua.RTBase` as input and registers an OPC UA namespace for the namespace stored in the RTBase class.

        Args:
            client : A client with a stored namespace.
        '''
        # get namespace from client
        client_namespace = client.client.namespace
        self.object_node_dict = {}
        # register a new namespace on the OPC UA server
        idx = self.opcua_server.register_namespace(client_namespace.namespace_name)

        # iterate through client namespace
        for namespace_entry in client_namespace.entry_list: 
            # check for every object node if it was already registered
            if namespace_entry.objectnode not in self.object_node_dict:
                # if not, register object node 
                object = self.opcua_server.nodes.objects.add_object(idx, namespace_entry.objectnode)
                self.object_node_dict[namespace_entry.objectnode] = object

            # add variable to object node
            self.add_variable_to_node(namespace_entry, idx)

        # write namespace index to client namespace for clients read/write methods
        client.client.add_namespace_url(idx)
        namespace_array = self.opcua_server.get_namespace_array()[2:]     
        print(f"The following namespaces are registered: {namespace_array}")


    # Add variables to a registered node
    def add_variable_to_node(self, namespace_entry:NamespaceEntry, namespace_url:int)->None:
        '''
        Adds a variable to a registered node on the OPC UA server.

        Args:
            namespace_entry : A OPCUA node ID. Contains the variable as well as the target node name.
            namespace_url : The namespace index identifying the namespace on the OPC UA server
        '''
        # create unique and descriptive variable name
        variable_name = f'{namespace_entry.objectnode}{namespace_entry.variable}'
        # add variable to object node
        datavector = self.object_node_dict[namespace_entry.objectnode].add_variable(
            opcua.ua.NodeId(variable_name, namespace_url), 
            variable_name, 
            [0.0]
            )
        
        datavector.set_writable()
        namespace_entry.variable = variable_name


    # Start server
    def start(self)->None:
        '''
        Starts the OPC UA server.
        '''
        try:
            self.opcua_server.start()

        except RuntimeError as err:
            print("The server "+ self.name +" could not be started, returned error message :\n", err)

    
        
    # Stop server
    def stop(self)->None:
        '''
        Stops the OPC UA server
        '''
        try:
            self.opcua_server.stop()
            print("The server  "+ self.name +" was stopped successfully @ ",time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))

        except RuntimeError as err:
            print("The server could not be stopped, returned error message :\n", err)

