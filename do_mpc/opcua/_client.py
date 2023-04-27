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
from typing import List
from casadi import *
from ._helper import Namespace, ClientOpts

try:
    import asyncua.sync as opcua
except ImportError:
    raise ImportError("The asyncua library is not installed. Please install it and try again.")

class RTClient:
    '''
    Real Time Client.
    The RTClient class extends do-mpc by an easy to setup OPC UA client.

    Note:

        The RTClient class main purpose is to setup an OPC UA client inside the :py:class:`do_mpc.opcua.RTBase` class.

    **Configuration and setup:**

    Configuring and setting up the RTClient client involves the following steps:

    1. Use :py:class:`do_mpc.opcua.ClientOpts` dataclass to specify client name as well as IP adress and port for the target server.

    2. Use the :py:class:`do_mpc.opcua.Namespace` dataclass to setup the namespace stored in the RTClient instance.

    3. Initiate the RTClient instance with instances of :py:class:`do_mpc.opcua.ClientOpts` and :py:class:`do_mpc.opcua.Namespace`.

    4. Connect the RTClient to the taget server with :py:meth:`connect`

    Note:

        Remember to disconnect the RTClient class afterwards with :py:meth:`disconnect`
    
    Args:
        opts : Client options.
        namespace : Namespace draft stored in RTClient.
    '''

    def __init__(self, opts:ClientOpts, namespace:Namespace)->None:
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

    # Method used by server to mark namespace with the corresponding url
    def add_namespace_url(self, url:int)->None:
        '''
        This method is used to add an OPC UA namespace index to the stored namespace.

        Args:
            url : The OPC UA namespace index.
        '''
        self.namespace._namespace_index = url

    # This function implements (re)connection to the designated server
    def connect(self)->None:
        '''
        Connects the client to the target server.
        '''
        try:
            self.opcua_client.connect()
            print("The -", self.name, "- has just connected to ",self.server_address)

        except RuntimeError:
            print("The connection to the server could not be established\n", self.server_address,
                  " is not responding")


    def disconnect(self)->None:
        '''
        Disconnects the client from the target server.
        '''
        self.opcua_client.disconnect()
        print("A client of type", self.name,"disconnected from server",self.server_address)
        

    def writeData(self, tag:str, dataVal:list)->None:
        '''
        Overwrites a variable on the target server.

        Args:
            tag : The node ID of the target variable on the OPC UA server.
            dataVal : The value written to the specified node ID
        '''
        assert type(dataVal) == list, "The data you provided is not arranged as a list. See the instructions for passing data to the server."
        
        try:
            self.opcua_client.get_node(tag).set_value(dataVal)
        except ConnectionRefusedError:
            print("Write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return False
        


    def readData(self, tag:str)->float:
        '''
        Reads a variable from the target server.

        Args:
            tag : The node ID of the target variable on the OPC UA server.
 
        Return:
            The value stored on the target server.
        '''         
        try: 
            dataVal = self.opcua_client.get_node(tag).get_value()
        except ConnectionRefusedError:
            print("A read operation by:", self.type, "failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        return dataVal