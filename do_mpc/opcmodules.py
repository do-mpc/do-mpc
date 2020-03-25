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
from casadi import *
from casadi.tools import *
import pdb
import pickle
import do_mpc
import opcua
import time
from do_mpc.simulator import Simulator
from do_mpc.estimator import *
from do_mpc.controller import MPC

class Server:
    """**do-mpc** OPCUA Server. An instance of this class is created for all active **do-mpc** classes,
    e.g. :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.estimator.EKF`.

    The class is initialized with relevant instances of the **do-mpc** configuration, e.g :py:class:`do_mpc.model.Model`, :py:class:`do_mpc.model.Optimizer` etc, which contain all
    information about variables (e.g. states, inputs, optimization outputs etc.).

    The :py:class:`Server` class has a public API, which can be used to manually create and launch a server from the interpreter. However, it is recommended to use the real-time manager object
    :py:class:`Manager`
    """
    def __init__(self, opts):
        self.dtype = 'default'
        model = opts['_model']
        #TODO: signal to the user that the server structure is initialized without the observer
       
        # The basic OPCUA server definition contains a name, address and a port numer
        # The user can decide if they want to activate the SQL database option (_with_db=TRUE)
        self.name    = opts['_name']
        self.address = opts['_address']
        self.port    = opts['_port']
        self.with_db = opts['_with_db']
        
        # The server type describes one of three levels of complexity: 
        # _server_type = 'basic' --> only necessary data structures are setup (functioning optimizer)
        # _server_type = 'with_parameters' --> also stores the [time varying] parameters of the model
        # _server_type = 'with_estimator'  --> if an MHE or EKF are implemented, the estimated states are stored
        # _server_Type = 'with_monitoring'  --> the server also stores additional KPIs for monitoring
        self.server_type = opts['_server_type']
        
        # If TRUE, the predictions of the optimizer are also stored 
        self.store_predictions = opts['_store_predictions']
        
        # TODO:fix number predictions
        #self._aux = np.empty((0, model.n_aux))
        # Dictionary with possible data_fields in the class and their respective dimension. All data is numpy ndarray.
        self.data_structure = {
            'nr_x_states': model.n_x,
            'nr_z_states': model.n_z,
            'nr_inputs'  : model.n_u,
            'nr_meas'    : model.n_y,
            'nr_controls': model.n_u,
            'nr_tv_pars' : model.n_tvp,
            'nr_mod_pars': model.n_p,
            'nr_aux'     : model.n_aux,
            'nr_pred'    : model.n_x * 20,
            'nr_flags'   : 5,
            'nr_switches': 5
        }
        # TODO: use the namespace to create the structure further down
        self.namespace = {
            'PlantData':{'x':"States.X",'z':"States.Z",'u':"Inputs",'y':"Measurements",'p':"Parameters"},
            'ControllerData':{'x0':"InitialGuess",'u_opt':"OptimalOutputs"},
            'EstimatorData':{'xhat':"Estimates.X",'phat':"Estimates.P"},
            'UserData':{'flags':"Flags",'switches':"Switches"}
            }
        try:
            self.opcua_server = opcua.Server()
        except RuntimeError:
            #TODO: add detailed error handling and inform user about possible actions
            self.created = False
            print("Server could not be created. Check your opcua module installation!")
            return False
        
        self.opcua_server.set_endpoint(self.address)
        self.opcua_server.set_server_name(self.name)
        
        # Setup a default namespace, because personalizing it does not bring any value for now
        idx = self.opcua_server.register_namespace("Realtime NMPC structure")

        # Get objects node, this is where the nodes are put
        objects = self.opcua_server.get_objects_node()
        
        # Create the basic data structure, which consists of the simulator and the optimizer
        localvar = objects.add_object(opcua.ua.NodeId("PlantData", idx), "PlantData")
        
        placeholder = [0 for x in range(self.data_structure['nr_x_states'])]
        datavector = localvar.add_variable(opcua.ua.NodeId("States.X", idx), "States.X", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_z_states'])] if self.data_structure['nr_z_states']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("States.Z", idx), "States.Z",  placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_meas'])] if self.data_structure['nr_meas']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Measurements", idx), "Measurements", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_inputs'])] if self.data_structure['nr_inputs']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Inputs", idx), "Inputs", placeholder)
        datavector.set_writable()
        if self.server_type == 'with_parameters':
            placeholder = [0 for x in range(self.data_structure['nr_mod_pars'])] if self.data_structure['nr_p']>0 else [0]
            datavector = localvar.add_variable(opcua.ua.NodeId("Parameters", idx), "Parameters", placeholder)
            datavector.set_writable()
        
        localvar = objects.add_object(opcua.ua.NodeId("ControllerData", idx), "ControllerData")
        
        placeholder = [0 for x in range(self.data_structure['nr_x_states'])] if self.data_structure['nr_x_states']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("InitialGuess", idx), "InitialGuess", placeholder)
        datavector.set_writable()
        placeholder = [0 for x in range(self.data_structure['nr_inputs'])] if self.data_structure['nr_inputs']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("OptimalOutputs", idx), "OptimalOutputs", placeholder)
        datavector.set_writable()
        if self.server_type == 'with_parameters':
            placeholder = [0 for x in range(self.data_structure['nr_tvp'])] if self.data_structure['nr_p']>0 else [0]
            datavector = localvar.add_variable(opcua.ua.NodeId("TVParameters", idx), "TVParameters",  placeholder)
            datavector.set_writable()
        if self.store_predictions == True:
            placeholder = [0 for x in range(self.data_structure['nr_pred'])] if self.data_structure['nr_pred']>0 else [0]
            datavector = localvar.add_variable(opcua.ua.NodeId("Predictions", idx), "Predictions", placeholder)
            datavector.set_writable()    
        
        if self.server_type == 'with_estimator':
            localvar = objects.add_object(opcua.ua.NodeId("EstimatorData", idx), "EstimatorData")
            placeholder = [0 for x in range(self.data_structure['nr_x_states'])]
            datavector = localvar.add_variable(opcua.ua.NodeId("Estimates.X", idx), "Estimates.X",  placeholder)
            datavector.set_writable()
            placeholder = [0 for x in range(self.data_structure['nr_mod_pars'])]
            datavector = localvar.add_variable(opcua.ua.NodeId("Estimates.P", idx), "Estimates.P",  placeholder)
            datavector.set_writable()
            
        # The flags are defined by default 
        localvar = objects.add_object(opcua.ua.NodeId("UserData", idx), "UserData")
        
        placeholder = [0 for x in range(self.data_structure['nr_flags'])] if self.data_structure['nr_flags']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Flags", idx), "Flags",  placeholder)
        datavector.set_writable()
        # The switches allow for manual control of the real-time modules remotely
        placeholder = [0 for x in range(self.data_structure['nr_switches'])] if self.data_structure['nr_switches']>0 else [0]
        datavector = localvar.add_variable(opcua.ua.NodeId("Switches", idx), "Switches", placeholder)
        datavector.set_writable()
        self.created = True
        self.running = False


    def start(self):

        try:
            self.opcua_server.start()

            print("Server was started @ ",time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            self.running = True
            return True
        except RuntimeError as err:
            print("Server could not be started, returned error message :\n", err)
            return False

    def stop(self):

        #TODO: stop services(optimizer, estimator) in order and inform user about success
        try:
            self.opcua_server.stop()

            print("Server was stopped successfully @ ",time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            self.running = False
            return True
        except RuntimeError as err:
            print("Server could not be stopped, returned error message :\n", err)
            return False

    def checkStatus(self):
        # This function returns status flags on the server
        #TODO: implement status flags and error checkup
        return True

    def update(self, **kwargs):
        """Update value(s) of the data structure with key word arguments.
        These key word arguments must exist in the data fields of the data objective.
        See self.data_fields for a complete list of data fields.

        Example:

        ::

            _name = "My basic OPCUA server"
            _port = 4880
            _mpc_model = model
            Server.update('_name': _name, '_port': _port, '_mpc_model': model)

            or:
            data.update('_name': _name)
            data.update('_port': _port)

            Alternatively:
            data_dict = {
                '_name':"My basic OPCUA server",
                '_port':"4880"
            }

            data.update(**data_dict)


        :param kwargs: Arbitrary number of key word arguments for data fields that should be updated.
        :type kwargs: OPCUA definition requires strings, while the data structure is derived from **do-mpc** objects

        :raises assertion: Keyword must be in existing data_fields.

        :return: None
        """
        for key, value in kwargs.items():
            assert key in self.data_fields.keys(), 'Cannot update non existing key {} in data object.'.format(key)
            if type(value) == structure3.DMStruct:
                value = value.cat
            if type(value) == DM:
                # Convert to numpy
                value = value.full()
            elif type(value) in [float, int, bool]:
                value = np.array(value)
            # Get current results array for the given key:
            arr = getattr(self, key)
            # Append current value to results array:
            updated = np.append(arr, value.reshape(1,-1), axis=0)
            # Update results array:
            setattr(self, key, updated)


class Client:
        """**do-mpc** OPCUA Client. An instance of this class is created for all active **do-mpc** classes,
        e.g. :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.estimator.MHE`.
    
        The class is initialized with relevant instances of the **do-mpc** configuration, e.g :py:class:`do_mpc.model.Model`, :py:class:`do_mpc.model.Optimizer` etc, which contain all
        information about variables (e.g. states, inputs, optimization outputs etc.).
    
        The :py:class:`Client` class has a public API, which can be used to manually create and launch a server from the interpreter. However, it is recommended to use the real-time manager object
        :py:class:`Manager`
        """
        def __init__(self, opts):
            self.server_address = opts['_address']
            self.port           = opts['_port']
            self.type           = opts['_client_type']
            self.namespace      = opts['_namespace']
            #self.namespace = {
            #'PlantData':{'x':"States.X",'z':"States.Z",'u':"Inputs",'y':"Measurements",'p':"Parameters"},
            #'ControllerData':{},
            #'EstimatorData':{}
            #'UserData':{'flags':"Flags",'switches':"Switches"}
            #}
            try:
                self.opcua_client = opcua.Client(self.server_address)
                print("A client of the type -", self.type, "- was created")
            except RuntimeError:
                # TODO: catch the correct error and parse message
                print("The connection to the server could not be established\n", self.server_address, "is not responding")

            self.created = True
            return
    
        def connect(self):
            # This function implements (re)connection to the designated server
            try:
                self.opcua_client.connect()
                print("The -", self.type, "- has just connected to ",self.server_address)
                self.connected = True
            except RuntimeError:
                # TODO: catch the correct error and parse message
                print("The connection to the server could not be established\n", self.server_address,
                      " is not responding")
    
        def disconnect(self):
            self.opcua_client.disconnect()
            self.connected = False
            print("A client of type", self.type,"disconnected from server",self.server_address)
            #TODO: disconnecting is done automatically upon console termination?
            
    
        def writeData(self, dataVal, tag):
            # This function can write data to any of the server defined namespaces
            # TODO: handle also type of writing e.g status data, operation flags etc
            assert type(dataVal) == list, "The data you provided is not arranged as a list. See the instructions for passing data to the server."
            assert "ns=2;s=" in tag, "The data destination you have provided is invalid. Refer to the OPCUA server namespace and define a correct source."
            
            try:
                wr_result = self.opcua_client.get_node(tag).set_value(dataVal)
            except ConnectionRefusedError:
                print("Write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
                return False
            
            return wr_result
    
        def readData(self, tag):
            # This function can read any data field from the server, from the source defined by the tag
            assert "ns=2;s=" in tag, "The data source you have provided is invalid. Refer to the OPCUA server namespace and define a correct source."
           
            try:
                dataVal = self.opcua_client.get_node(tag).get_value()
            except ConnectionRefusedError:
                print("Read operation by:", self.type, "failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return dataVal


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
        self.enabled    = True
        self.iter_count = 0
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        self.feedback = do_mpc.estimator.StateFeedback(model)
        
        self.opc_client.connect()
        
        # The server must be initialized with the x0 values of the simulator
        tag = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        self.opc_client.writeData(self._x0.cat.toarray().tolist(), tag)
    
    def init_server(self, dataVal):
         # The server must be initialized with the x0 values of the simulator
        tag = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time simulator could not connect to the server. Please correct the server setup.")
        
        
    def stop(self):
        try:
            self.opc_client.disconnect()
            self.enabled = False
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The real-time simulator could not be stopped due to server issues.")
              
    def run_asynchronously(self):
        """
        This function implements the server calls and simulator step with a predefined frequency
        :param no params: because the cycle is stored by the object
        
        :return: none
        :rtype: none
        """
        while self.enabled:
            #Read the latest control inputs from server and execute step
            uk = np.array(self.opc_client.readData())
            
            yk = self.make_step(uk)
            xk = self.feedback.make_step(yk)
            
            # The full state vector is written back to the server
            self.opc_client.writeData(xk.tolist())
            
            # The simulator must wait for a predefined time
            time.sleep(self.cycle_time)
            
    def asynchronous_step(self):
        
        tag_in  = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
        tag_out = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        #Read the latest control inputs from server and execute step
        uk = np.array(self.opc_client.readData(tag_in))
        
        yk = self.make_step(uk)
        xk = self.feedback.make_step(yk)
        
        # The full state vector is written back to the server
        self.opc_client.writeData(xk.tolist(), tag_out)
        
        self.iter_count = self.iter_count + 1

     
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
        
        try:
            self.opc_client.connect()
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The real-time controller could not connect to the server.")
            self.is_ready = False
            
        # The server must be initialized with the x0 values of the simulator
        if self.opc_client.connected:
            tag = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
            self.opc_client.writeData(model._u(0).cat.toarray().tolist(), tag)
            self.is_ready = True
        else:
            print("The server data could not be initialized by the controller!")
            self.is_ready = False 
            
    def init_server(self, dataVal):
         # The server must be initialized with the u0 values of the simulator
        tag = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time controller could not connect to the server. Please correct the server setup.")
        # One optimizer iteration is called before starting the cyclical operation
        self.asynchronous_step()

    def stop(self):
        try:
            self.opc_client.disconnect()
            self.enabled = False
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The real-time controller could not be stopped due to server issues.")
                
    def check_status(self):
        """
        This function is called before every optimization step to ensure that the server data
        is sane and a call to the optimizer can be made in good faith, i.e. the plant state
        contains meaningful data and that no flags have been raised.
        
        :param no params: this function onyl needs internal data
        
        :return: check_result is the result of all the check done by the controller before executing the step
        :rtype: bool
        """
        check_result = self.is_ready
        # Step 1: check that the server is running
        check_result = self.opc_client.connected and check_result
        # Step 2: check whether the controller should run and no controller flags have been raised
        
        # Step 2: check that the plant/simulator is running and no simulator flags have been raised
        
        self.is_ready = check_result
        return check_result
    
    def _initialize_optimizer(self):
        """
        This is an internal function meant to be called before each controller step to reinitialize the initial guess of 
        the NLP solver with the most recent plant data.
        
        :param no params: because the cycle is stored by the object
        
        :return: none
        :rtype: none
        """
        x0 = np.array(self.opc_client.readData())
        # TODO: implement a more "clever" initialization, e.g by running the integrator over the prediction horizon
        
    def run_asynchronously(self):
        """
        This function implements the server calls and simulator step with a predefined frequency
        :param no params: because the cycle is stored by the object
        
        :return: none
        :rtype: none
        """
        while self.enabled:
            #Read the latest plant state from server and execute optimization step
            xk = np.array(self.opc_client.readData())
            
            uk = self.make_step(xk)
            
            # The optimal inputs are written back to the server
            self.opc_client.writeData(uk.tolist())
            
            # The controller must wait for a predefined time
            time_left = self.cycle_time-self.solver_stats['t_wall_S']
            if time_left>0: time.sleep(time_left)
            
    def asynchronous_step(self):
        """
        This function implements the server calls and simulator step with a predefined frequency
        :param no params: because the cycle is stored by the object
        
        :return: time_left, the remaining time on the clock when the optimizer has finished the routine
        :rtype: float
        """
        if self.output_feedback:
            tag_in  = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        else:
            tag_in  = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        tag_out = "ns=2;s="+self.opc_client.namespace['ControllerData']['u_opt']
        
        # Read the latest plant state from server and execute optimization step
        xk = np.array(self.opc_client.readData(tag_in))
        
        # The NLP must be reinitialized with the most current data from the plant readings
        self.set_initial_state(np.array(self.opc_client.readData(tag_in)), reset_history=True)
        self.set_initial_guess()

        # Check the current status before running the optimizer step 
        if self.check_status():
            uk = self.make_step(xk)
        else: 
            print("The controller failed executing the step @ ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        # The optimal inputs are written back to the server
        self.opc_client.writeData(uk.tolist(), tag_out)
        
        self.iter_count = self.iter_count + 1
        # The controller must wait for a predefined time
        time_left = self.cycle_time-self.solver_stats['t_wall_S']
       
        return time_left


class RealtimeFeedback(StateFeedback):
    """The basic real-time, asynchronous estimator, which expands on the ::do-mpc class Estimator.
    This class implements an asynchronous operation, making use of a connection to a predefined OPCUA server, as
    a means of exchanging information with other modules, e.g. an NMPC controller or a simulator.
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
        assert opts['_opc_opts']['_client_type'] == 'estimator', "You must define this module with an estimator OPC Client. Please review the opts dictionary."

        super().__init__(model)
        self.enabled    = True
        self.iter_count = 0
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        
        try:
            self.opc_client.connect()
        except RuntimeError:
            self.enabled = False
            print("The real-time estimator could not connect to the server. Please correct the server setup.")
            
        # The server must be initialized with the x0 values of the simulator
        tag = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        self.opc_client.writeData(self._x0.cat.toarray().tolist(), tag)
    
    def init_server(self, dataVal):
         # The server must be initialized with the x0 values of the simulator
        tag = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        try:
            self.opc_client.writeData(dataVal, tag)
        except RuntimeError:
            self.enabled = False
            print("The real-time estimator could not connect to the server. Please correct the server setup.")
        
        
    def stop(self):
        try:
            self.opc_client.disconnect()
            self.enabled = False
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The real-time estimator could not be stopped due to server issues.")
            
            
    def asynchronous_step(self):
        """
        This function implements the server calls and estimator step with a predefined frequency
        :param no params: all parameters are stored by the object
        
        :return: time_left, represents the leftover time until the max cycle time of the estimator
        :rtype: none
        """
        
        #Read the latest plant state from server and execute optimization step
        tag_in  = "ns=2;s="+self.opc_client.namespace['PlantData']['x']
        tag_out = "ns=2;s="+self.opc_client.namespace['EstimatorData']['xhat']
        tic = time.time()
        xk = np.array(self.opc_client.readData(tag_in))
        
        xk_hat = self.make_step(xk)
        
        # The optimal inputs are written back to the server
        self.opc_client.writeData(xk_hat.tolist(), tag_out)
        toc = time.time()
        # The controller must wait for a predefined time
        time_left = self.cycle_time-(toc-tic)
        
        self.iter_count = self.iter_count + 1
        return time_left


from threading import Timer

class RealtimeTrigger(object):
    """
    This class is employed in timing the execution of your real-time ::do-mpc modules. One RealtimeTrigger is required 
    for every module, i.e. one for the simulator, one for the controller and one for the estimator, if the latter is present.
    """
        
    def __init__(self, interval, function, *args, **kwargs):
        """
        This function implements the server calls and simulator step with a predefined frequency
        :interval: the cycle time in seconds representing the frquency with which the target function is executed
        :function: a function to be called cyclically
        :args : arguments to pass to the target function
        :return: none
        :rtype: none
        """
        self._timer = None
        self.interval = interval
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.is_running = False
        self.next_call = time.time()
        self.start()
    
    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)
  
    def start(self):
        if not self.is_running:
            self.next_call += self.interval
            self._timer = Timer(self.next_call - time.time(), self._run)
            self._timer.start()
            self.is_running = True
  
    def stop(self):
        self._timer.cancel()
        self.is_running = False