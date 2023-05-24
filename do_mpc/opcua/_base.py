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
from typing import List
from threading import Timer, Thread
from enum import Enum, auto
from casadi import *
from ._client import RTClient
from ._helper import Namespace, NamespaceEntry, ClientOpts
import casadi.tools as ctools
from ..model import Model

   
class RTBase:
    ''' Real Time Base.
    The RTBase class extends do-mpc with an OPC UA interface.

    Note:

        The :py:class:`do_mpc.estimator.MHE` class is currently not supported.

    Use this class to configure an OPC UA client for a previously initiated do-mpc class e.g.. :py:class:`do_mpc.controller.MPC` or :py:class:`do_mpc.simulator.Simulator`.

    **Configuration and setup:**

    Configuring and setting up the RTBase class involves the following steps:

    1. Use :py:class:`do_mpc.opcua.ClientOpts` dataclass to specify client name as well as IP adress and port number of the target server.

    2. Initiate the RTBase class with a do-mpc object and the dataclass :py:class:`do_mpc.opcua.ClientOpts`.

    3. Use :py:meth:`set_write_tags` and :py:meth:`set_read_tags` to take over the namespace tags (node IDs) from another instance of RTBase (optional).

    Note:

        Use :py:meth:`set_write_tags` and :py:meth:`set_read_tags` only after registering all clients with the :py:meth:`do_mpc.opcua.RTServer.namespace_from_client` method.


    4. Use :py:meth:`connect` to connect the client to the OPC UA server.

    4. Use :py:meth:`write_to_tags` to write initial values to the OPC UA server.

    5. Use :py:meth:`async_step_start` to run the do-mpc method :py:meth:`do_mpc.controller.MPC.make_step`.

    Args:
        do_mpc_object : An instance of a do-mpc class.
        clientOpts : Client Options
        namespace : Namespace containing OPC UA node IDs
    '''

  

    def __init__(self, do_mpc_object, clientOpts: ClientOpts, namespace:Namespace=None)->None:
        
        self.do_mpc_object = do_mpc_object

        if namespace == None:
            self.get_default_namespace(clientOpts.name)
        else:
            self.namespace = namespace

        self.cycle_time = do_mpc_object.settings.t_step*clientOpts.timeunit
        self.client = RTClient(clientOpts, self.namespace)
        self.tagout = []
        self.tagin = []
        self.is_running = False
        self.new_init = True
        self.async_fag = False


    def namespace_from_model(self, model:Model, model_name:str)->Namespace:

        """
        Create a OPC UA namespace from the provided model.
        
        Args:
            model : A do-mpc model.
            model_name : Name given to the generated namespace.

        Returns:
            Namespace generated from the OPC UA model.


        """

        node_list = []
        variable_list = ['aux', 'p', 'tvp', 'u', 'v', 'w', 'x', 'y', 'z']

        for var in variable_list:
            
            for key in model[var].labels():
                if 'default' in key:
                    continue
                
                node_list.append(NamespaceEntry(var, key))

        return Namespace(model_name, node_list)


    def get_default_namespace(self, namespace_name:str)->None:
        '''
        Sets default namespace using :py:meth:`namespace_from_model`.
        
        Args:
            namespace_name : Name given to the generated namespace
        '''
        self.namespace = self.namespace_from_model(self.do_mpc_object.model, namespace_name)


    def connect(self)->None:
        '''
        Connects client to the server.
        '''

        try:
            self.client.connect()
        except RuntimeError:
            self.enabled = False


    def disconnect(self)->None:
        '''
        Disconnects client from the server.
        '''

        try:
            self.client.disconnect()
        except RuntimeError:
            print("The real-time controller could not be stopped due to server issues. Please stop the client manually and delete the object!")


    def set_write_tags(self, tagout:List[str])->None:
        '''
        Set tags (node IDs) to write to. The provided tags must match the node IDs registered on the taget server.

        Args:
            tagout : A list of node IDs to which the client writes.
        '''
        self.tagout = tagout



    def set_read_tags(self, tagin:List[str])->None:
        '''
        Set tags (node IDs) to read from. The provided tags must match the node IDs registered on the taget server.
        
        Args:
            tagin : A list of node IDs from which the client reads.

        '''
        self.tagin = tagin


    def make_step(self)->None:
        '''
        Calls the do-mpc make_step method e.g.. :py:meth:`do_mpc.controller.MPC.make_step`.
        The input for make_step is taken from node IDs specified in :py:meth:`read_from_tags`.
        The output of make_step is written to the node IDs specified in :py:meth:`write_to_tags`.
        '''
        input = self.read_from_tags()
        output = self.do_mpc_object.make_step(input)
        self.write_to_tags(output)


    def write_to_tags(self, data:np.ndarray)->None:
        '''
        Write to the node IDs specified in :py:meth:`write_to_tags`

        Args:
            data : data which is written to server.
 
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


    def read_from_tags(self)->np.ndarray:
        '''
        Read from the node IDs specified in :py:meth:`read_from_tags`.

        Returns:
            Values stored on the OPC UA server.
        '''
        return np.array([self.client.readData(i) for i in self.tagin]).reshape(-1,1)


    def async_run(self)->None:
        '''
        This method is called inside of :py:meth:`async_step_start`. It calls :py:meth:`make_step` and :py:meth:`async_step_start`.
        '''
        self.is_running = False    
        self.async_step_start()
        self.make_step()


    def async_step_start(self)->None:
        '''
        This method calls the :py:meth:`async_run` method in a frequency given by the do-mpc classes t_step value.
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

        if self.async_fag == True:
            self.async_fag = False
            return print('Async operation was interrupted by the user')


    def async_step_stop(self)->None:
        '''
        Stops :py:meth:`async_step_start` from running.
        '''
        self.thread.cancel()
        self.is_running = False
        self.new_init = True
        self.async_fag = True



