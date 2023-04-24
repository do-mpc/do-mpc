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
from dataclasses import dataclass
from typing import List
from threading import Timer, Thread
from casadi import *
from ._client import RTClient
import casadi.tools as ctools

        
class RTBase:
    ''' Real Time Base.
    The RTBase class extends the :py:class:`do_mpc.controller.MPC` and the :py:class:`do_mpc.Simulator` classes with an OPC UA interface.

    Use this class to configure a OPC UA client with previously initiated :py:class:`do_mpc.controller.MPC` or :py:class:`do_mpc.Simulator` instances.

    **Configuration and setup:**

    Configuring and setting up the MPC controller involves the following steps:

    1. Use :py:class:`do_mpc.opcua_wrapper.ClientOpts` dataclass to specify client name as well as IP adress and port of the target server.

    2. Initiate the RTBase class with a do-mpc object and the dataclass :py:class:`do_mpc.opcua_wrapper.ClientOpts`.

    3. Use :py:meth:`set_write_tags` and :py:meth:`set_read_tags` to take over the tags from another instance of RTBase (optional).

    4. Use :py:meth:`connect` to connect the clienet to the OPC UA server.

    4. Use :py:meth:`write_to_tags` to write the initial values to the OPC UA server.

    5. Use :py:meth:`async_step_start` to run the do-mpc :py:meth:`do_mpc.MPC.make_step`.

    .. note::

        This class currently doesent work with the :py:class:`do_mpc.estimator.MHE` class.
    
    '''

    def __init__(self, do_mpc_object, clientOpts, namespace=None):
        
        """
        Initializes the real-time base object with the provided do_mpc object, client options, and an optional namespace.
        
        :param do_mpc_object: The initiated instance of a do-mpc class. Currently supported are :py:class:`do_mpc.controller.MPC` nd :py:class:`do_mpc.Simulator`
        :type: class

        :param clientOpts: A dataclass specifying the client name , IP address and port for the target server. See :py:class:`do_mpc.opcua_wrapper.ClientOpts` for further information.
        :dtype: dataclass

        :param namespace: A dataclass specifying tags for the OPC UA namespace (optional). If no namespace dataclass is given, a default namespace is generated from the do-mpc object. See :py:class:`do_mpc.opcua_wrapper.Namespace` for further information.
        :type: dataclass
        """

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


    def namespace_from_model(self, model, model_name):

        """
        Create a namespace from the provided model.
        
        :param model: A do-mpc model :py:class:`do_mpc.model.Model`
        :type model: class

        :param model_name: Name given to the generated namespace.
        :type model_name: str

        """

        node_list = []
        variable_list = ['aux', 'p', 'tvp', 'u', 'v', 'w', 'x', 'y', 'z']

        for var in variable_list:
            
            for key in model[var].labels():
                if 'default' in key:
                    continue
                
                node_list.append(NamespaceEntry(var, key))

        return Namespace(model_name, node_list)


    def get_default_namespace(self, namespace_name):
        '''
        Set default namespace using :py:meth:`namespace_from_model`.
        '''
        self.def_namespace = self.namespace_from_model(self.do_mpc_object.model, namespace_name)


    def connect(self):
        '''
        Connect client to server.
        '''

        try:
            self.client.connect()
        except RuntimeError:
            self.enabled = False


    def disconnect(self):
        '''
        Disconnect client from server.
        '''

        try:
            self.client.disconnect()
        except RuntimeError:
            print("The real-time controller could not be stopped due to server issues. Please stop the client manually and delete the object!")


    def set_write_tags(self,tagout:List[str]):
        '''
        Set tags to write to. The provided tags must match the variable names registered on the taget server.

        :param tagout: A list of strings specifying the variables the client is writing.
        :type tagout: List[str]
        '''
        self.tagout = tagout



    def set_read_tags(self, tagin:List[str]):
        '''
        Set tags to read from. The provided tags must match the variable names registered on the taget server.

        :param tagout: A list of strings specifying the variables the client is reading from.
        :type tagout: List[str]
        '''
        self.tagin = tagin


    def make_step(self):
        '''
        The do-mpc make_step method e.g. :py:meth:`do_mpc.MPC.make_step`
        '''
        input = self.read_from_tags()
        output = self.do_mpc_object.make_step(input)
        self.write_to_tags(output)


    def write_to_tags(self, data):
        '''
        Write results of :py:meth:`make_step` to server.

        :param data: data which is written to server.
        :type data: ctools.structure3.DMStruct, ctools.DM, np.ndarray
        '''
        if isinstance(data, ctools.structure3.DMStruct):
            data = data.cat.full().flatten()
        elif isinstance(data, ctools.DM):
            data = data.full().flatten()
        elif isinstance(data, np.ndarray):
            data = data.flatten()
        else:
            raise TypeError(f'Unsupported dtype:{type(data)}')

        if data.size != len(self.tagout):
            raise Exception(f'Trying to write {len(data)} elements to {len(self.tagout)}') 
        

        for tag, value in zip(self.tagout, data):
            self.client.writeData(tag, [value])


    def read_from_tags(self):
        '''
        Read current values from server.
        '''
        return np.array([self.client.readData(i) for i in self.tagin]).reshape(-1,1)


    def async_run(self):
        self.is_running = False    
        self.async_step_start()
        self.make_step()


    def async_step_start(self):
        '''
        Call the :py:meth:`make_step` method in a frequency given by the do-mpc classes t_step value.
        '''
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
        '''
        Stop :py:meth:`async_step_start` from running.
        '''
        self.thread.cancel()
        self.is_running = False
        self.new_init = True



@dataclass
class NamespaceEntry:
    objectnode: str
    variable: str


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