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

from dataclasses import dataclass
from typing import List

@dataclass
class NamespaceEntry:
    '''
    An OPC UA node ID Namespace Entry.
    A helper class to create an OPC UA node ID for :py:class:`do_mpc.opcua.Namespace`.

    Args:
        objectnode : Object node name .
        variable : Variable name.
    '''
    objectnode: str
    variable: str


    def get_node_id(self, namespace_index:int)->str:
        '''
        Creates a node ID containing the namespace index as well as the variable name.

        Args:
            namespace_index : A OPC UA namespace index.

        Return:
            An OPC UA node ID string containing the namespace index and the variable name.

        '''

        if namespace_index == None:
            raise Exception('Namespace_index not defined')
        return f'ns={namespace_index};s={self.variable}'
    


@dataclass   
class Namespace:
    '''
    An OPC UA Namespace draft.
    A helper class to create node IDs for the setup of an OPC UA namespace. Used to setup :py:class:`do_mpc.opcua.RTBase` and :py:class:`do_mpc.opcua.RTClient`.

    Args:
        namespace_name : Namespace name.
        entry_list : A list of node IDs.
        _namespace_index : The index of an OPC UA namespace.
    '''
    namespace_name: str
    entry_list: List[NamespaceEntry]
    _namespace_index: int = None

    def __getitem__(self, nodename: str):
        return [entry.get_node_id(self._namespace_index) for entry in self.entry_list if entry.objectnode == nodename]



@dataclass
class ServerOpts:
    '''
    Server Options.
    A helper class to correctly define server options. Used for the setup of :py:class:`do_mpc.opcua.RTServer` 

    Args:
        name : Name of the server.
        address : IP address of the server.
        port : Used port number.
    '''
    name: str
    address: str
    port: int


@dataclass
class ClientOpts:
    '''
    Client Options.
    A helper class to correctly define client options. Used for the setup of :py:class:`do_mpc.opcua.RTClient`.

    Args:
        name : Name of the client.
        address : IP address of the target server.
        port : Used port number of the target server.
        timeunit: Time unit factor to convert the time unit used by the dynamic system into seconds. The default value is 1 for seconds. Use 60 for minutes, 3600 for hours, and so on. 
    '''

    """
    Name of the client.
    """
    name: str
    """
    IP address of the target server.
    """

    address: str
    port: int
    timeunit: int = 1