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

import numpy as np
from casadi import *
from casadi.tools import *
import pdb


class simulator:
    """[Summary]

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)
    :raises [ErrorType]: [ErrorDescription]
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    def __init__(self, model):
        self.model = model

    def set_param(self, integration_tool=None, integrator_options=None, model_type=None):
        """[Summary]

        :param integration_tool: , defaults to None
        :type [ParamName]: [ParamType](, optional)
        :raises [ErrorType]: [ErrorDescription]
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        if integration_tool:
            self.integration_tool = integration_tool
        if integrator_options:
            self.integrator_options = integrator_options
        if model_type:
            self.model_type = model_type


    def check_validity(self):
        """Function to check if all necessary values for simulating a model are given.

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        :raises [ErrorType]: [ErrorDescription]
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        pass

    def simulate(self,_x=None,_u=None,_p=None,_tvp=None,_z=None):
        # TODO: check in model if some variables are empty, and if given or not
        """Function to check if all necessary values for simulating a model are given.

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        :raises [ErrorType]: [ErrorDescription]
        :return: Returns
        :rtype: [ReturnType]
        """
        pdb.set_trace()
        # states
        if _x:
            _x = _x
        else:
            _x = self._x

        # inputs
        if _u:
            _u = _u

            if _p:
                _p = _p

        if self.model.model_type == 'discrete':
            self._x = self.simulator(_x,_u,_p,_tvp,_z)
        elif self.model.model_type = 'continuous':
            self._x = self.simulator(x0 = 0, p = 0)
