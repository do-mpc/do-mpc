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
import time
from do_mpc.estimator import StateFeedback, EKF, MHE
from casadi import *
try:
    import opcua
except ImportError:
    raise ImportError("The opcua library is not installed. Please install it and try again.")
 
from .opcua_client import *
       

class RealtimeEstimator():
    """The basic real-time, asynchronous estimator, which expands on the ::python:class:`do-mpc.Estimator`. The inheritance is done selectively,
    which means that this class prototype can inherit from any of the available estimators. The selection is done at runtime and must be specified by the user
    through a type variable (see the constructor of the class). This class implements an asynchronous operation, making use of a connection to a predefined OPCUA server, as
    a means of exchanging information with other modules, e.g. an NMPC controller or a simulator.
    """
    def __new__(cls, etype,  model, opts):
        etype = {'state-feedback': StateFeedback, 'EKF': EKF, 'MHE': MHE}[etype]
        cls = type(cls.__name__ + '+' + etype.__name__, (cls, etype), {})
        return super(RealtimeEstimator, cls).__new__(cls)
    
    def __init__(self, etype, model, opts):
        """The constructor of the class. Creates a real-time estimator and sets the parameters.
        
        :param etype: The base estimator type to inherit from. Available types are `state-feedback`=::py:class:`do_mpc:Estimator:StateFeedback`,
        `EKF`=::py:class:`do_mpc:Estimator:EKF` and `MHE`=::py:class:`do_mpc:Estimator:MHE`
        :type etype: string
        
        :param model: A **do-mpc** model which contains the data used to initialize the estimator
        :type model: ::py:class:`do_mpc:Model`
        
        :opts: a dictionary of parameters, mainly cycle_time and data structure of the server
        :type opts: cycle_time: float
                    opc_opts: dict, see Client settings

        :return: None
        :rtype: None
        """
        assert opts['_opc_opts']['_client_type'] == 'estimator', "You must define this module with an estimator OPC Client. Please review the opts dictionary."

        super().__init__(model)
        self.etype      = etype
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
            print("The real-time estimator could not connect to the server. Please check the server setup.")
            
        # The server must be initialized with the x0 values of the simulator
        tag = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        self.opc_client.writeData(self._x0.cat.toarray().tolist(), tag)
    
    def init_server(self):
        """Initializes the OPC-UA server with the initial estimator values (states and parameters). If the operation does
        not succeed, the estimator is deemed unable to carry on and the `self.enable` attribute is set to `False`.
        """
        tag = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        dataVal = np.array(vertcat(self.x0)).tolist()
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time estimator could not connect to the server. Please correct the server setup.")
    
    def start(self):
        """Alternative method to start the client from the console. The client is usually automatically connected upon instantiation.
        
        :return result: The result of the connection attempt to the OPC-UA server.
        :rtype result: boolean
        """
        try:
            self.opc_client.connect()
            self.enabled = True
        except RuntimeError:
            self.enabled = False
            print("The real-time estimator could not connect to the server. Please check the server setup.")
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
            # TODO: catch the correct error and parse message
            print("The real-time estimator could not be stopped because the connection to the server was interrupted. Please stop the client manually and delete the object!")
        return self.enabled
            
    def asynchronous_step(self):
        """This function implements one server call and estimator step, after which it writes output data back to the server.
        
        :param no params: all parameters are stored by the underlying estimator object
        
        :return: time_left, represents the leftover time until the max cycle time of the estimator
        :rtype: none
        """
        
        #Read the latest plant data from server 
        tag_in_u  = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
        tag_in_x  = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        tag_in_y  = "ns=2;s="+self.opc_client.namespace['PlantData']['y']
        tag_in_p  = "ns=2;s="+self.opc_client.namespace['PlantData']['p']
        tag_out_x = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        tag_out_p = "ns=2;s="+self.opc_client.namespace['EstimatorData']['phat']
        
        tic = time.time()
        
        if self.etype == 'state-feedback':           
            xk = np.array(self.opc_client.readData(tag_in_x))
            xk_hat = self.make_step(xk)
            # The latest estimates are written back to the server
            self.opc_client.writeData(xk_hat.tolist(), tag_out_x)
        
        if self.etype == 'EKF':            
            uk = np.array(self.opc_client.readData(tag_in_u))
            yk = np.array(self.opc_client.readData(tag_in_y))
            pk = np.array(self.opc_client.readData(tag_in_p))
            xk_hat, pk_hat = self.make_step(yk,uk,pk)
            
            # The latest estimates are written back to the server
            self.opc_client.writeData(xk_hat.tolist(), tag_out_x)
            try:
                self.opc_client.writeData(pk_hat.tolist(), tag_out_p)
            except RuntimeError:
                print("Cannot write parameter estimates because the server refused the write operation")
        
        if self.etype == 'MHE':
            raise NotImplementedError('MHE is currently not implemented in RealTme do-MPC')
            # TODO: the call is yet to be implemente din real-time fashion
            xk_hat = self._x0
            self.opc_client.writeData(xk_hat.tolist(), tag_out_x)
            
        toc = time.time()
        # The estimator must wait for a predefined time
        time_left = self.cycle_time-(toc-tic)
        
        self.iter_count = self.iter_count + 1
        return time_left