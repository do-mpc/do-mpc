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
from do_mpc.estimator import Estimator
from do_mpc.controller import MPC

class Server:
    """**do-mpc** OPCUA Server. An instance of this class is created for all active **do-mpc** classes,
    e.g. :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.estimator.MHE`.

    The class is initialized with relevant instances of the **do-mpc** configuration, e.g :py:class:`do_mpc.model.Model`, :py:class:`do_mpc.model.Optimizer` etc, which contain all
    information about variables (e.g. states, inputs, optimization outputs etc.).

    The :py:class:`Server` class has a public API, which can be used to manually create and launch a server from the interpreter. However, it is recommended to use the real-time manager object
    :py:class:`Manager`
    """
    def __init__(self, opts):
        self.dtype = 'default'
        model = opts['_model']
        assert model.flags['setup'] == True, 'Model was not setup. After the complete model creation call model.setup_model().'
        #assert optimizer.flags['setup'] == True, 'Optimizer was not setup. After the complete model creation call model.setup_model().'
        #assert simulator.flags['setup'] == True, 'Simulator was not setup. After the complete model creation call model.setup_model().'
        #TODO: signal to the user that the server structure is initialized without the observer
        #assert observer.flags['setup'] == True, 'Observer was not setup. After the complete model creation call model.setup_model().'

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
            'EstimatorData':{'flags':"Flags",'switches':"Switches"},
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
            placeholder = [0 for x in range(self.data_structure['nr_p'])] if self.data_structure['nr_p']>0 else [0]
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
        
        if self.server_type == 'with_estimatior':
            localvar = objects.add_object(opcua.ua.NodeId("EstimatorData", idx), "EstimatorData")
            datavector = localvar.add_variable(opcua.ua.NodeId("States", idx), "EstimatedStates",  placeholder)
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
            
    
        def writeData(self, dataVal):
            # This function can write data to any of the server defined namespaces
            # TODO: handle also type of writing e.g status data, operation flags etc
            assert type(dataVal) == list, "The data you provided is not arranged as a list. See the instructions for passing data to the server."
            if self.type == "simulator":
                plant_update = "ns=2;s="+self.namespace['PlantData']['x']
                contr_update = "ns=2;s="+self.namespace['ControllerData']['x0']
            if self.type == "controller":
                plant_update = "ns=2;s="+self.namespace['PlantData']['u']
                contr_update = "ns=2;s="+self.namespace['ControllerData']['u_opt']
            try:
                out1 = self.opcua_client.get_node(plant_update).set_value(dataVal)
                out2 = self.opcua_client.get_node(contr_update).set_value(dataVal)
            except ConnectionRefusedError:
                print("Write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
                return False
            return (out1 and out2)
    
        def readData(self):
            # This function can read any data field from the server
    
            dataVal = []
            if self.type == "simulator":
                readVar = "ns=2;s="+self.namespace['PlantData']['u']
            if self.type == "controller":
                readVar = "ns=2;s="+self.namespace['ControllerData']['x0']
            try:
                dataVal = self.opcua_client.get_node(readVar).get_value()
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
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        self.feedback = do_mpc.estimator.StateFeedback(model)
        
        self.opc_client.connect()
        
        # The server must be initialized with the x0 values of the simulator
        self.opc_client.writeData(self._x0.cat.toarray(1).tolist())
        
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
            
    def run_once(self):
        #Read the latest control inputs from server and execute step
        uk = np.array(self.opc_client.readData())
        
        yk = self.make_step(uk)
        xk = self.feedback.make_step(yk)
        
        # The full state vector is written back to the server
        self.opc_client.writeData(xk.tolist())
        
        # The simulator must wait for a predefined time
        time.sleep(self.cycle_time)
        
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
        self.cycle_time = opts['_cycle_time']
        self.opc_client = Client(opts['_opc_opts'])
        
        
        self.opc_client.connect()
        
        # The server must be initialized with the x0 values of the simulator
        self.opc_client.writeData(model._u(0).cat.toarray(1).tolist())
        
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
            
    def run_once(self):
        """
        This function implements the server calls and simulator step with a predefined frequency
        :param no params: because the cycle is stored by the object
        
        :return: none
        :rtype: none
        """
        
        #Read the latest plant state from server and execute optimization step
        xk = np.array(self.opc_client.readData())
        
        uk = self.make_step(xk)
        
        # The optimal inputs are written back to the server
        self.opc_client.writeData(uk.tolist())
        
        # The controller must wait for a predefined time
        time_left = self.cycle_time-self.solver_stats['t_wall_S']
        if time_left>0: time.sleep(time_left)