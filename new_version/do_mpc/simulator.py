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
    """A class for simulating systems. Discrete-time and continuous systems can be considered.
    """
    def __init__(self, model):
        """ Initialize the simulator class. The model gives the basic model description and is used to build the simulator. If the model is discrete-time, the simulator is a function, if the model is continuous, the simulator is an integrator.

        :param model: Simulation model
        :type var_type: model class

        :return: None
        :rtype: None
        """
        self.model = model

        assert model.flags['setup'] == True, 'Model for simulator was not setup. After the complete model creation call model.setup_model().'

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

    def set_initial_state(self, x0, reset_history=False):
        """Set the intial state of the simulator.
        Optionally resets the history. The history is empty upon creation of the simulator.

        :param x0: Initial state
        :type x0: numpy array
        :param reset_history: Resets the history of the simulator, defaults to False
        :type reset_history: bool (optional)

        :return: None
        :rtype: None
        """
        assert x0.size == self.model._x.size, 'Intial state cannot be set because the supplied vector has the wrong size. You have {} and the model is setup for {}'.format(x0.size, self.model._x.size)
        assert isinstance(reset_history, bool), 'reset_history parameter must be of type bool. You have {}'.format(type(reset_history))
        if isinstance(x0, (np.ndarray, casadi.DM)):
            self._x0 = self.model._x(x0)
        elif isinstance(x0, structure3.DMStruct):
            self._x0 = x0
        else:
            raise Exception('x0 must be of tpye (np.ndarray, casadi.DM, structure3.DMStruct). You have: {}'.format(type(x0)))

        if reset_history:
            self.reset_history()

    def reset_history(self):
        """Reset the history of the simulator
        """
        self.data.init_storage()


    def setup_simulator(self):
        """Sets up the simulator after the parameters were set via set_param. The simulation time step is required in order to setup the simulator for continuous and discrete-time models.

        :raises assertion: t_step must be set

        :return: None
        :rtype: None
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
        ])

        self.sim_p_num = self.sim_p(0)

        if self.model.model_type == 'discrete':

            # Build the rhs expression with the newly created variables
            x_next = self.model._rhs_fun(sim_x['_x'],sim_p['_u'],sim_x['_z'],sim_p['_tvp'],sim_p['_p'])

            # Build the simulator function
            self.simulator = Function('simulator',[sim_x,sim_p],[x_next])

        elif self.model.model_type == 'continuous':

            # Define the ODE
            xdot = self.model._rhs_fun(sim_x['_x'],sim_p['_u'],sim_x['_z'],sim_p['_tvp'],sim_p['_p'])
            dae = {
                'x': sim_x['_x'],
                'z': sim_x['_z'],
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

        sim_aux = self.model._aux_expression_fun(sim_x['_x'],sim_p['_u'],sim_x['_z'],sim_p['_tvp'],sim_p['_p'])
        # Create function to caculate all auxiliary expressions:
        self.sim_aux_expression_fun = Function('sim_aux_expression_fun', [sim_x, sim_p], [sim_aux])


    def set_param(self, **kwargs):
        """Set the parameters for the simulator. Setting the simulation time step t_step is necessary for setting up the simulator via setup_simulator.

        :param integration_tool: Sets which integration tool is used, defaults to cvodes (only continuous)
        :type integration_tool: string
        :param abstol: gives the maximum allowed absolute tolerance for the integration, defaults to 1e-10 (only continuous)
        :type abstol: float
        :param reltol: gives the maximum allowed relative tolerance for the integration, defaults to 1e-10 (only continuous)
        :type abstol: float
        :param t_step: Sets the time step for the simulation
        :type t_step: float

        :return: None
        :rtype: None
        """

        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for {} model.'.format(key, self.model.model_type))
            setattr(self, key, value)

        # if any settings were changed for a continuous model, update the simulator
        if any([kw in self.data_fields for kw in kwargs.keys()]) and (self.model.model_type == 'continuous'):
            self.setup_simulator()


    def get_tvp_template(self):
        """Obtain the a numerical copy of the structure of the time-varying parameters for the simulation.

        :return: numerical CasADi structure
        :rtype: struct_SX
        """
        return self.model._tvp(0)


    def set_tvp_fun(self,tvp_fun):
        """Function to set the function which gives the values of the time-varying parameters.

        :param tvp_fun: [ParamDescription], defaults to [DefaultParamVal]
        :type tvp_fun: python function

        :raises [ErrorType]: the output of tvp_fun must have the right structure

        :return: None
        :rtype: None
        """
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        self.tvp_fun = tvp_fun


    def get_p_template(self):
        """Obtain the a numerical copy of the structure of the parameters for the simulation.

        :return: numerical CasADi structure
        :rtype: struct_SX
        """
        return self.model._p(0)


    def set_p_fun(self,p_fun):
        """Function to set the function which gives the values of the parameters.

        :param p_fun: A function which gives the values of the parameters
        :type p_fun: python function

        :raises assert: p must have the right structure

        :return: None
        :rtype: None
        """
        assert self.get_p_template().labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun


    def simulate(self):
        """This is the core function of the simulator class. Numerical values for sim_x_num and sim_p_num need to be provided beforehand in order to simulate the system for one time step:

        * states (sim_x_num['_x'])

        * algebraic states (sim_x_num['_z'])

        * inputs (sim_p_num['_u'])

        * parameter (sim_p_num['_p'])

        * time-varying parameters (sim_p_num['_tvp'])

        The function returns the new state of the system.

        :return: x_new
        :rtype: numpy array
        """
        # extract numerical values
        sim_x_num = self.sim_x_num
        sim_p_num = self.sim_p_num

        if self.model.model_type == 'discrete':
            x_new = self.simulator(sim_x_num,sim_p_num)
            z_now = self.sim_x_num['_z']
        elif self.model.model_type == 'continuous':
            r = self.simulator(x0 = sim_x_num, p = sim_p_num)
            x_new = r['xf']
            z_now = r['zf']
        aux_now = self.sim_aux_expression_fun(sim_x_num, sim_p_num)

        self.sim_x_num = self.sim_x(vertcat(x_new,z_now))
        self.sim_aux_num = self.model._aux_expression(aux_now)

        return x_new

    def make_step(self, u0, x0=None, z0=None):
        """Main method of the simulator class during control runtime. This method is called at each timestep
        and returns the next state for the current control input ``u0``.
        The initial state ``x0`` is stored as a class attribute but can optionally be supplied.
        The algebraic states ``z0`` can also be supplied, if they are defined in the model but are only used as an intial guess.

        The method prepares the simulator by setting the current parameters, calls :py:func:`simulator.simulate`
        and updates the :py:class:`do_mpc.data` object.

        :param u0: Current input to the system.
        :type u0: numpy.ndarray

        :param x0: Current state of the system.
        :type x0: numpy.ndarray (optional)

        :param z0: Initial guess for current algebraic states
        :type z0: numpy.ndarray (optional)

        :return: x_nsext
        :rtype: numpy.ndarray
        """
        assert isinstance(u0, (np.ndarray, casadi.DM, structure3.DMStruct)), 'u0 is wrong input type. You have: {}'.format(type(u0))
        assert u0.shape == self.model._u.shape, 'u0 has incorrect shape. You have: {}, expected: {}'.format(u0.shape, self.model._u.shape)
        assert isinstance(u0, (np.ndarray, casadi.DM, structure3.DMStruct)), 'u0 is wrong input type. You have: {}'.format(type(u0))
        assert u0.shape == self.model._u.shape, 'u0 has incorrect shape. You have: {}, expected: {}'.format(u0.shape, self.model._u.shape)

        if x0 is None:
            x0 = self._x0
        else:
            assert isinstance(x0, (np.ndarray, casadi.DM, structure3.DMStruct)), 'x0 is wrong input type. You have: {}'.format(type(x0))
            assert x0.shape == self.model._x.shape, 'x0 has incorrect shape. You have: {}, expected: {}'.format(x0.shape, self.model._x.shape)

        if z0 is not None:
            assert isinstance(z0, (np.ndarray, casadi.DM, structure3.DMStruct)), 'z0 is wrong input type. You have: {}'.format(type(z0))
            assert z0.shape == self.model._z.shape, 'z0 has incorrect shape. You have: {}, expected: {}'.format(z0.shape, self.model._z.shape)
            # Just an initial guess.
            self.sim_x_num['_z'] = z0

        tvp0 = self.tvp_fun(self._t0)
        p0 = self.p_fun(self._t0)
        t0 = self._t0
        self.sim_x_num['_x'] = x0
        self.sim_p_num['_u'] = u0
        self.sim_p_num['_p'] = p0
        self.sim_p_num['_tvp'] = tvp0

        self.simulate()

        x_next = self.sim_x_num['_x']
        z0 = self.sim_x_num['_z']
        aux0 = self.sim_aux_num

        self.data.update(_x = x0)
        self.data.update(_u = u0)
        self.data.update(_z = z0)
        self.data.update(_tvp = tvp0)
        self.data.update(_p = p0)
        self.data.update(_aux_expression = aux0)
        self.data.update(_time = t0)

        self._x0 = x_next
        self._t0 = self._t0 + self.t_step

        return x_next.full()
