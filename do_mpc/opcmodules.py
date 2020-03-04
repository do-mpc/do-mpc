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
        
        # TODO: n_aux not existing
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
        localvar = objects.add_object(opcua.ua.NodeId("SimulatorData", idx), "SimulatorData")
        datavector = localvar.add_variable(opcua.ua.NodeId("States.X", idx), "States.X", self.data_structure['nr_x_states'])
        datavector.set_writable()
        datavector = localvar.add_variable(opcua.ua.NodeId("States.Z", idx), "States.Z", self.data_structure['nr_z_states'])
        datavector.set_writable()
        datavector = localvar.add_variable(opcua.ua.NodeId("Measurements", idx), "Measurements", self.data_structure['nr_meas'])
        datavector.set_writable()
        datavector = localvar.add_variable(opcua.ua.NodeId("Inputs", idx), "Inputs", self.data_structure['nr_inputs'])
        datavector.set_writable()
        if self.server_type == 'with_parameters':
            datavector = localvar.add_variable(opcua.ua.NodeId("Parameters", idx), "Parameters", self.data_structure['nr_x_states'])
            datavector.set_writable()
        
        localvar = objects.add_object(opcua.ua.NodeId("OptimizerData", idx), "OptimizerData")
        datavector = localvar.add_variable(opcua.ua.NodeId("InitialGuess", idx), "InitialGuess", self.data_structure['nr_x_states'])
        datavector.set_writable()
        datavector = localvar.add_variable(opcua.ua.NodeId("OptimalOutputs", idx), "OptimalOutputs", self.data_structure['nr_inputs'])
        datavector.set_writable()
        if self.server_type == 'with_parameters':
            datavector = localvar.add_variable(opcua.ua.NodeId("TVParameters", idx), "TVParameters", self.data_structure['nr_tv_pars'])
            datavector.set_writable()
        if self.store_preditions == True:
            datavector = localvar.add_variable(opcua.ua.NodeId("Predictions", idx), "Predictions", self.data_structure['nr_x_states'])
            datavector.set_writable()    
        
        if self.server_type == 'with_estimatior':
            localvar = objects.add_object(opcua.ua.NodeId("EstimatorData", idx), "EstimatorData")
            datavector = localvar.add_variable(opcua.ua.NodeId("States", idx), "States", self.data_structure['nr_x_states'])
            datavector.set_writable()
            
        # The flags are defined by default 
        localvar = objects.add_object(opcua.ua.NodeId("User", idx), "User")
        datavector = localvar.add_variable(opcua.ua.NodeId("Flags", idx), "Flags", self.data_structure['nr_flags'])
        datavector.set_writable()
        # The switches allow for manual control of the real-time modules remotely
        datavector = localvar.add_variable(opcua.ua.NodeId("Switches", idx), "Switches", self.data_structure['nr_switches'])
        datavector.set_writable()
        self.created = True
        self.running = False


    def start(self):

        try:
            self.opcua_server.start()

            print("Server was started successfully")
            self.running = True
            return True
        except RuntimeError as err:
            print("Server could not be started, returned error message :\n", err)
            return False

    def stop(self):

        #TODO: stop services(optimizer, estimator) in order and inform user about success
        try:
            self.opcua_server.stop()

            print("Server was stopped successfully")
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
    
        def __init__(self, opts):
            self.server_address = opts['_address']
            self.port           = opts['_port']
            self.type           = opts['_type']
            
            try:
                self.opcua_client = opcua.Client(self.server_address)
                print("A client of the type", self.type, "was created")
            except RuntimeError:
                # TODO: catch the correct error and parse message
                print("The connection to the server could not be established\n", self.settings["server"], "is not responding")

        self.created = True
        return
    
    def connect(self):
        # This function implements (re)connection to the designated server
        try:
            self.opcua_client.connect()
            print("A client of the type", self.settings["type"], "was created")
            self.status["connected"] = True
        except RuntimeError:
            # TODO: catch the correct error and parse message
            print("The connection to the server could not be established\n", self.settings["server"],
                  " is not responding")
            return False

        return True

    def disconnect(self):
        self.opcua_client.disconnect()
        print("A client of type", self.settings["type"],"disconnected from server",self.settings["server"])
        #TODO disconnecting is by default done
        return

    def writeData(self, dataType, dataVal):
        # This function can write data to any of the server defined namespaces
        # TODO: handle also typed of writing e.g status data, operation flags etc

        if self.type == "simulator":
            writevar = "ns=2;s=States.vector"

        if self.type == "optimizer":
            writevar = "ns=2;s=Inputs.vector"
        try:
            out = self.opcua_client.get_node(writevar).set_value(dataVal)
        except ConnectionRefusedError:
            print("Write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return False
        return out

    def readData(self, dataType):
        # This function can read any data from the server

        dataVal = []
        if self.type == "simulator":
            readVar = "ns=2;s=Inputs.vector"
        if self.type == "optimizer":
            readVar = "ns=2;s=States.vector"
        try:
            dataVal = self.opcua_client.get_node(readVar).get_value()
        except ConnectionRefusedError:
            print("Read operation by:", self.type, "failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        return dataVal