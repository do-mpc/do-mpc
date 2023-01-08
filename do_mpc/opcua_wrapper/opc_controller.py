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
from do_mpc.controller import MPC
from casadi import *
try:
    import opcua
except ImportError:
    raise ImportError("The opcua library is not installed. Please install it and try again.")

from .opcua_client import *


class RealtimeController(MPC):
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
        assert opts['_opc_opts']['_client_type'] == 'controller', "You must define this module with a controller OPC Client. Review your opts dictionary."

        super().__init__(model)
        self.enabled    = True
        self.iter_count = 0
        self.output_feedback = opts['_output_feedback']
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        self.user_controlled = opts['_user_controlled']
        
        try:
            self.opc_client.connect()
        except RuntimeError:
            print("The real-time controller could not connect to the server.")
            self.is_ready = False
            
        # The server must be initialized with the nonzero values for the inputs and the correct input structure
        if self.opc_client.connected:
            tag = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
            self.opc_client.writeData(model._u(0).cat.toarray().tolist(), tag)
            self.is_ready = True
        else:
            print("The server data could not be initialized by the controller!")
            self.is_ready = False 
            
    def init_server(self):
        """Initializes the OPC-UA server with the first MPC values for the optimal inputs to prevent simulator or estimator crashes.
        If the operation does not succeed, the controller is deemed unable to carry on and the `self.enable` attribute is set to `False`.
        
        :param dataVal: the first optimal input vector at time t=0
        :type dataVal: list of float
        
        :return result: The result of server write operation
        :rtype result: boolean
        """
        tag = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
        dataVal = np.array(vertcat(self.u0)).flatten().tolist()
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time controller could not connect to the server. Please correct the server setup.")
        
        # One optimizer iteration is called before starting the cyclical operation
        
        self.asynchronous_step()
        
        return self.enabled
    
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
            print("The real-time controller could not connect to the server. Please check the server setup.")
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
            print("The real-time controller could not be stopped due to server issues. Please stop the client manually and delete the object!")
        return self.enabled
    
    def check_status(self):
        """This function is called before every optimization step to ensure that the server data
        is sane and a call to the optimizer can be made in good faith, i.e. the plant state
        contains meaningful data and that no flags have been raised.
        
        :param no params: this function onyl needs internal data
        
        :return: check_result is the result of all the check done by the controller before executing the step
        :rtype: boolean
        """
        check_result = self.is_ready
        # Step 1: check that the server is running and the client is connected
        check_result = self.opc_client.connected and check_result
        if check_result == False: 
            print("The controller check failed because: controller not connected to server.")
            return False
        
        # Step 2: check whether the user has requested to run the optimizer
        if self.user_controlled:
            check_result = check_result and self.opc_client.checkSwitches(pos=0)
            if check_result == False: 
                print("The controller check failed because: controller not manually enabled on the server.")
                return False
        
        # Step 3: check whether the controller should run and no controller flags have been raised
        # flags = [0-controller, 1-simulator, 2-estimator, 3-monitoring, 4-extra]
        check_result = check_result and not self.opc_client.checkFlags(pos=0)
        if check_result == False: 
            print("The controller check failed because: controller has raised a failure flag.")
            return False
        
        # Step 4: check that the plant/simulator is running and no simulator flags have been raised
        check_result = check_result and  not (self.opc_client.checkFlags(pos=1) or self.opc_client.checkFlags(pos=2))
        self.is_ready = check_result
        if check_result == False: 
            print("The controller check failed because: either the simulator or estimator have reported crap data. Unsafe to run the controller!")
        return check_result
    
    # def initialize_optimizer(self, style='static'):
    #     """This is an internal function meant to be called before each controller step to reinitialize the initial guess of 
    #     the NLP solver with the most recent plant data.
        
    #     :param style: The type of initialization to be performed. A choice of three options is available. `static` denotes a crude 
    #                 initialization using the current state values over the entire prediction horizon. `dynamic` uses an integrator 
    #                 to simulate from the current state, over the prediction horizon, using the current inputs. Finally `predictive`
    #                 uses a one-step lookahead to pre-initialize the optimizer considering the average delay time.
    #     :type style: string in [`static`, `dynamic`, `predictive`]
        
    #     :return: none
    #     :rtype: none
    #     """
    #     if style == 'static':
    #         # This is a 'crude' initialization, which copies the same value over the entire prediction horizon
    #         x0 = np.array(self.opc_client.readData())
    #     if style == 'dynamic':
    #         # This is a dynamic initialization
    #         x0 = np.array(self.opc_client.readData())
    #     if style == 'predictive':
    #         # TODO: implement the step ahead prediction based on the cycle time delays
    #         x0 = np.array(self.opc_client.readData())
    
    #     # The NLP must be reinitialized with the most current data from the plant readings
    #     self.set_initial_state(x0, reset_history=True)
    #     self.set_initial_guess()
        
    def asynchronous_step(self):
        """This function implements the server calls and simulator step with a predefined frequency
        :param no params: because the cycle is stored by the object
        
        :return: time_left, the remaining time on the clock when the optimizer has finished the routine
        :rtype: float
        """
        if self.output_feedback == False:
            tag_in  = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        else:
            tag_in  = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        
        tag_out = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
        # tag_out_pred = "ns=2;s="+self.opc_client.namespace['ControllerData']['x_pred']
        # Read the latest plant state from server and execute optimization step
        xk = np.array(self.opc_client.readData(tag_in))
        
        # The NLP must be reinitialized with the most current data from the plant readings
        self.x0 = np.array(self.opc_client.readData(tag_in))   
        self.set_initial_guess()

        # Check the current status before running the optimizer step 
        if self.check_status():
            # The controller can be executed
            uk = self.make_step(xk)
            x_pred = self.opt_x_num_unscaled
            # The iteration count is incremented regardless of the outcome
            self.iter_count = self.iter_count + 1
            
            if self.solver_stats['return_status'] == 'Solve_Succeeded':
                # The optimal inputs are written back to the server
                self.opc_client.writeData(uk.tolist(), tag_out)
                # self.opc_client.writeData(np.array(vertcat(x_pred)).tolist(), tag_out_pred)
            else:
                print("The controller failed at time ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
                print("The optimal inputs have not been updated on the server.")
            # The controller must wait for a predefined time
            time_left = self.cycle_time-self.solver_stats['t_wall_total']
            
        else: 
            time_left = self.cycle_time
            print("The controller is still waiting to be manually activated. When you're ready set the status bit to 1.")
        
        return time_left