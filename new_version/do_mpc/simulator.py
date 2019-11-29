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
import do_mpc.data


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
        self.data = do_mpc.data.model_data(model)

        self._x0 = model._x(0)
        self._t0 = np.array([0])

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
            entry('_z', struct=self.model._z),
        ])

        self.sim_x_num = self.sim_x(0)

        self.sim_p = sim_p = struct_symSX([
            entry('_u', struct=self.model._u),
            entry('_p', struct=self.model._p),
            entry('_tvp', struct=self.model._tvp),
            entry('_z', struct=self.model._z)
        ])

        self.sim_p_num = self.sim_p(0)

        if self.model.model_type == 'discrete':

            # Build the rhs expression with the newly created variables
            x_next = self.model._rhs_fun(sim_x['_x'],sim_p['_u'],sim_x['_z'],sim_p['_tvp'],sim_p['_p'])

            # Build the simulator function
            self.simulator = Function('simulator',[sim_x,sim_p],[x_next])

        elif self.model.model_type == 'continuous':

            # Define the ODE
            xdot = self.model._rhs_fun(sim_x['_x'],sim_p['_u'],sim_p['_z'],sim_p['_tvp'],sim_p['_p'])
            dae = {
                'x': sim_x['_x'],
                'z': sim_z['_z'],
                'p': sim_p,
                'ode': xdot
            }

            # Set the integrator options
            opts = {
                'abstol': self.abstol,
                'reltol': self.reltol,
                'tf': self.t_step
            }

            # Build the simulator
            self.simulator = integrator('simulator', self.integration_tool, dae,  opts)


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


    def get_tvp_template(self):
        return self.model._tvp(0)


    def set_tvp_fun(self,tvp_fun):
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        self.tvp_fun = tvp_fun


    def get_p_template(self):
        return self.model._p(0)


    def set_p_fun(self,tvp_fun):
        assert self.get_p_template().labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun


    def simulate(self):
        # TODO: check in model if some variables are empty, and if given or not
        """Function to check if all necessary values for simulating a model are given.

        :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
        :type [ParamName]: [ParamType](, optional)
        :raises [ErrorType]: [ErrorDescription]
        :return: Returns
        :rtype: [ReturnType]
        """

        # extract numerical values
        sim_x_num = self.sim_x_num
        sim_p_num = self.sim_p_num

        if self.model.model_type == 'discrete':
            x_new = self.simulator(sim_x_num,sim_p_num)
        elif self.model.model_type == 'continuous':
            x_new = self.simulator(x0 = sim_x_num, p = sim_p_num)['xf']

        return x_new
