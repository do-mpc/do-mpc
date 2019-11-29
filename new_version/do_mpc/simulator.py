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
        # TODO: set default values for integrator
        self.data_fields = [
            't_step'
        ]

        if self.model.model_type == 'continuous':

            # expand possible data fields
            self.data_fields.extend([
                'abstol',
                'reltol',
                'integration_tool'
            ])

            # set default values
            self.abstol = 1e-10
            self.reltol = 1e-10
            self.integration_tool = 'cvodes'


    def setup_simulator(self):
        """Sets up simulator
        """
        # Check if simulation time step was given
        assert self.t_step, 't_step is required in order to setup the simulator. Please set the simulation time step via set_param(**kwargs)'

        self.sim_x = sim_x = struct_symSX([
            entry('_x', struct=self.model._x),
        ])

        self.sim_p = sim_p = struct_symSX([
            entry('_u', struct=self.model._u),
            entry('_p', struct=self.model._p),
            entry('_tvp', struct=self.model._tvp),
            entry('_z', struct=self.model._z)
        ])

        if self.model.model_type == 'discrete':

            # Build the rhs expression with the newly created variables
            x_next = self.model._rhs_fun(sim_x['_x'],sim_p['_u'],sim_p['_z'],sim_p['_tvp'],sim_p['_p'])

            # Build the simulator function
            self.simulator = Function('simulator',[sim_x['_x'],sim_p['_u'],sim_p['_z'],sim_p['_tvp'],sim_p['_p']],[x_next])

        elif self.model.model_type == 'continuous':

            # Define the ODE
            dae = {
                'x': sim_x,
                'p': sim_p,
                'ode': self.model._rhs
            }

            # Set the integrator options
            opts = {
                'abstol': self.abstol,
                'reltol': self.reltol,
                'tf': self.t_step
            }

            # Build the simulator
            simulator = integrator('simulator', self.integration_tool, dae,  opts)


    def set_param(self, **kwargs):
        """[Summary]

        :param integration_tool: , defaults to None
        :type [ParamName]: [ParamType](, optional)
        :raises [ErrorType]: [ErrorDescription]
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        # TODO: Check for continuous system
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for {} model.'.format(key, self.model.model_type))
            setattr(self, key, value)

        # if any settings were changed for a continuous model, update the simulator
        if any([kw in self.data_fields for kw in kwargs.keys()]) and (self.model.model_type == 'continuous'):
            self.setup_simulator()


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
        elif self.model.model_type == 'continuous':
            self._x = self.simulator(x0 = 0, p = 0)
