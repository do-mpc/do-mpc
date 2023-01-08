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


import time

try:
    import opcua
except ImportError:
    raise ImportError("The opcua library is not installed. Please install it and try again.")
        

class Client:
    """**do-mpc** OPCUA Client. An instance of this class is created for all active **do-mpc** classes,
    e.g. :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.estimator.EKF` etc.

    The class is initialized with relevant instances of the **do-mpc** configuration, e.g :py:class:`do_mpc.model.Model`, :py:class:`do_mpc.model.Optimizer` etc, which contain all
    information about variables (e.g. states, inputs, optimization outputs etc.).

    """
    def __init__(self, opts):
        self.server_address = opts['_address']
        self.port           = opts['_port']
        self.type           = opts['_client_type']
        self.namespace      = opts['_namespace']

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
        

    def writeData(self, dataVal, tag):
        """ Writes a tag onto the OPCUA server. It is used to write MPC data (states, parameters and inputs) and is called by methods
            from any of the real-time MPC modules. Asserts if the string has the right format and whether the data is a list.
            
            :param dataVal: a list of data to be wtitten on a tag
            :type dataVal: list
            
            :param tag: a name representing a valid tag on the server to which the client is connected.
            :type tag: string
            
            :return wr_result: The writing result as returned by the OPCUA writing method
            :rtype wr_result: boolean
        """
        assert type(dataVal) == list, "The data you provided is not arranged as a list. See the instructions for passing data to the server."
        assert "ns=2;s=" in tag, "The data destination you have provided is invalid. Refer to the OPCUA server namespace and define a correct source."
        
        try:
            wr_result = self.opcua_client.get_node(tag).set_value(dataVal)
        except ConnectionRefusedError:
            print("Write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return False
        
        return wr_result

    def readData(self, tag):
        """ Reads values from a tag on the OPCUA server. It is used to read MPC data (states, parameters and inputs) and is called by methods
            from any of the real-time MPC modules. Asserts if the string has the right format and throws an error is the read operation has failed.
            
            :param tag: a name representing a valid tag on the server to which the client is connected.
            :type tag: string
            
            :return dataVal: The data list read from the server
            :rtype dataVal: list
            """
        assert "ns=2;s=" in tag, "The data source you have provided is invalid. Refer to the OPCUA server namespace and define a correct source."
       
        try:
            dataVal = self.opcua_client.get_node(tag).get_value()
        except ConnectionRefusedError:
            print("A read operation by:", self.type, "failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        return dataVal

    def checkFlags(self, pos=-1):
        """ Checks the controller related flags on the server and returns the overall state of the controller. The result can be either
        'OK' if there is nothing wrong with the optimizer or 'ERROR', when the optimization finished with an error flag. By default, any 
        flag that does not mean that the optimization step has finished successfully will be treated as a fault. Please consult
        the Ipopt exit flags to further debug the error.
        
        :param none
        :type none
        
        :return result: A string containing the result of the flag checks.
        :rtype result: string
        """
        assert pos<5 and pos>-2, "Please provide a flag position in the interval [-1,4]"
        
        tag = "ns=2;s=Flags"
        try:
            dataVal = self.opcua_client.get_node(tag).get_value()
        except ConnectionRefusedError:
            print("A read operation attempted by the base client failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        if pos == -1: 
            result = dataVal
        else:
            result =  dataVal[pos]
        return result
    
    def updateFlags(self, pos=3, flagVal=[1]):
        """ Writes the flags  Asserts if the string has the right format and whether the data is a list.
            
            :param flagVal: a list of data to be wtitten on a tag
            :type flagVal: list
            
            :param pos: position of the flag to be checked. If =-1, all flags will be returned
            :type pos: int
            
            :return wr_result: The writing result as returned by the OPCUA writing method
            :rtype wr_result: boolean
        """
        assert type(flagVal) == list, "The flag data you provided is not arranged as a list. See the instructions for passing data to the server."
        assert pos<5 and pos>-1, "Please provide a flag position in the interval [-1,4]"
        
        tag = "ns=2;s=Flags"
        if pos == -1: dataVal = flagVal
        else:
            dataVal = self.opcua_client.get_node(tag).get_value()
            dataVal[pos] = flagVal
        try:
            wr_result = self.opcua_client.get_node(tag).set_value(dataVal)
        except ConnectionRefusedError:
            print("A write operation by the base client failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return False
        return wr_result
    
    def checkSwitches(self, pos=-1):
        """ Reads the status of the real-time modules from the OPC-UA server. If pos=-1 will return the entire status vector,
        otherwise will return only the value at the required position. 
        
        :param pos: position in the check vector between [-1, 4]
        :type pos: integer
        
        :return result: the value read from the server
        :rtype result: list of integer
        """
        tag = "ns=2;s=Switches"
        try:
            dataVal = self.opcua_client.get_node(tag).get_value()
        except ConnectionRefusedError:
            print("A status read operation attempted by the base client failed @ time: ", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
        if pos == -1: 
            result = dataVal
        else:
            result =  dataVal[pos]
        return result
    
    '''
    updateSwitches() is not working in its current state, nor is it used by any of the classes/methods.
    '''
    
    def updateSwitches(self, pos=3, switchVal=1):
        """ Updates the status of the real-time modules on the OPC-UA server. If pos=-1, it writes the entire status vector,
        otherwise will update only the value at the required position. 
        
        :param pos: position in the status vector between [-1, 4]
        :type pos: integer
        
        :return result: success status of the status write operation
        :rtype status: boolean
        """
        # assert type(switchVal) == list, "The flag data you provided is not arranged as a list. See the instructions for passing data to the server."
        assert pos<4 and pos>-2, "Please provide a flag position in the interval [-1,4]"
        
        tag = "ns=2;s=Switches"
        if pos == -1: dataVal = switchVal
        else:
            dataVal = self.opcua_client.get_node(tag).get_value()
            dataVal[pos] = switchVal
        try:
            wr_result = self.opcua_client.get_node(tag).set_value(dataVal)

        except ConnectionRefusedError:
            print("A write operation by:", self.type, " failed @ time:", time.strftime('%Y-%m-%d %H:%M %Z', time.localtime()))
            return False
        return wr_result