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

import numpy as np
from do_mpc.simulator import Simulator
from casadi import *
try:
    import opcua
except ImportError:
    raise ImportError("The opcua library is not installed. Please install it and try again.")

from .opcua_client import *


class RealtimeSimulator(Simulator):
    """The basic real-time, asynchronous simulator, which expands on the ::do-mpc class Simulator.
    This class implements an asynchronous operation, making use of a connection to a predefined OPCUA server, as
    a means of exchanging information with other modules, e.g. an NMPC controller or an estimator.
    """
    def __init__(self, model, opts):
        """
        
        :param model: Initial state
        :type model: numpy array
        :opts: a dictionary of parameters, mainly cycle_time and data structure of the server
        :type opts: cycle_time: float
                    opc_opts: dict, see Client settings

        :return: None
        :rtype: None
        """
        assert opts['_opc_opts']['_client_type'] == 'simulator', "You must define this module with asimulator OPC Client. Review your opts dictionary."

        super().__init__(model)
        self.enabled    = False       
        self.iter_count = 0
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        self.user_controlled = opts['_user_controlled']
        
        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time simulator could not connect to the server. Please check the server setup.")

        # The server must be initialized with the x0 and p0 values of the simulator
        tag = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        self.opc_client.writeData(self._x0.cat.toarray().tolist(), tag)

    
    def init_server(self):
        """Initializes the OPC-UA server with the first plant values. The simulator is typcially the one that starts first and writes state and output values.
        If the operation does not succeed, the simulator is deemed unable to carry on and the `self.enable` attribute is set to `False`.
        
        :param dataVal: the first optimal input vector at time t=0
        :type dataVal: list of float
        
        :return result: The result of server write operation
        :rtype result: boolean
        """
        tag = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        dataVal = np.array(vertcat(self.x0)).tolist()
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time simulator could not connect to the server. Please correct the server setup.")

        
    def start(self):
        """Alternative method to start the simulator from the console. The client is usually automatically connected upon instantiation.
        
        :return result: The result of the connection attempt to the OPC-UA server.
        :rtype result: boolean
        """
        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time simulator could not connect to the server. Please check the server setup.")
        return self.enabled
        
    def stop(self):
        """ Stops the execution of the real-time estimator by disconnecting the OPC-UA client from the server. 
        Throws an error if the operation cannot be performed.
        
        :return result: The result of the disconnect operation
        :rtype: boolean
        """
        try:
            self.opc_client.disconnect()
            self.enabled = False
        except RuntimeError:
            print("The real-time simulator could not be stopped due to server issues. Please stop the client manually and delete the object!")
        return self.enabled

    def asynchronous_step(self):
        """ This function implements the server calls and simulator step. It must be used in combination 
            with a real-time trigger of the type ::py:class:`RealtimeTrigger`, which calls this routine with a predefined frequency. 
            The starting and stopping are also controlled via the trigger object.
        
        :param no params: because all information is stored in members of the object
        
        :return: none
        :rtype: none
        """
        
        tag_in    = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
        tag_out_x = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        tag_out_y = "ns=2;s="+self.opc_client.namespace['PlantData']['y']
        tag_out_p = "ns=2;s="+self.opc_client.namespace['PlantData']['p']
        
        #Read the latest control inputs from server and execute step
        uk = np.array(self.opc_client.readData(tag_in))
        
        yk = self.make_step(uk)
        xk = self._x0.cat.toarray() 
        # The parameters can't be read using p_fun because the internal timestamp is not relevant 
        pk = self.sim_p_num['_p'].toarray()
        
        # The full state vector is written back to the server (mainly for debugging and tracing)
        self.opc_client.writeData(xk.tolist(), tag_out_x)
        # The measurements are written to the server and will be used by the estimators
        self.opc_client.writeData(yk.tolist(), tag_out_y)
        # And also the current parameters, which are needed e.g by the estimator
        self.opc_client.writeData(pk.tolist(), tag_out_p)
        
        self.iter_count = self.iter_count + 1