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
import warnings
from do_mpc.data import Data
import do_mpc.model

class Simulator(do_mpc.model.IteratedVariables):
    """A class for simulating systems. Discrete-time and continuous systems can be considered.

    **do-mpc** uses the CasADi interface to popular state-of-the-art tools such as Sundials `CVODES`_
    for the integration of ODE/DAE equations.

    .. _CVODES: https://computing.llnl.gov/projects/sundials

    **Configuration and setup:**

    Configuring and setting up the simulator involves the following steps:

    1. Set parameters with :py:func:`set_param`, e.g. the sampling time.

    2. Set parameter function with :py:func:`get_p_template` and  :py:func:`set_p_fun`.

    3. Set time-varying parameter function with :py:func:`get_tvp_template` and  :py:func:`set_tvp_fun`.

    4. Setup simulator with :py:func:`setup`.

    During runtime, call the simulator :py:func:`make_step` method with current input (``u``).
    This computes the next state of the system and the respective measurement.
    Optionally, pass (sampled) random variables for the process ``w`` and measurement noise ``v`` (if they were defined in :py:class`do_mpc.model.Model`)

    """
    def __init__(self, model):
        """ Initialize the simulator class. The model gives the basic model description and is used to build the simulator. If the model is discrete-time, the simulator is a function, if the model is continuous, the simulator is an integrator.

        :param model: Simulation model
        :type var_type: model class

        :return: None
        :rtype: None
        """
        self.model = model
        do_mpc.model.IteratedVariables.__init__(self)

        assert model.flags['setup'] == True, 'Model for simulator was not setup. After the complete model creation call model.setup().'

        self.data = Data(model)

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

        self.flags = {
            'set_tvp_fun': False,
            'set_p_fun': False,
            'setup': False,
        }


    def reset_history(self):
        """Reset the history of the simulator.
        """
        self._t0 = np.array([0])
        self.data.init_storage()

    def _check_validity(self):
        # tvp_fun must be set, if tvp are defined in model.
        if self.flags['set_tvp_fun'] == False and self.model._tvp.size > 0:
            raise Exception('You have not supplied a function to obtain the time-varying parameters defined in model. Use .set_tvp_fun() prior to setup.')
        # p_fun must be set, if p are defined in model.
        if self.flags['set_p_fun'] == False and self.model._p.size > 0:
            raise Exception('You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() prior to setup.')

        # Set dummy functions for tvp and p in case these parameters are unused.
        if not self.flags['set_tvp_fun']:
            _tvp = self.get_tvp_template()
            def tvp_fun(t): return _tvp
            self.set_tvp_fun(tvp_fun)

        if not self.flags['set_p_fun']:
            _p = self.get_p_template()
            def p_fun(t): return _p
            self.set_p_fun(p_fun)

        assert self.t_step, 't_step is required in order to setup the simulator. Please set the simulation time step via set_param(**kwargs)'


    def setup(self):
        """Sets up the simulator and finalizes the simulator configuration.
        Only after the setup, the :py:func:`make_step` method becomes available.

        :raises assertion: t_step must be set

        :return: None
        :rtype: None
        """

        self._check_validity()

        self.sim_x = sim_x =  self.model.sv.sym_struct([
            entry('_x', struct=self.model._x)
            ])
        self.sim_z = sim_z =  self.model.sv.sym_struct([
            entry('_z', struct=self.model._z)
            ])

        self.sim_p = sim_p = self.model.sv.sym_struct([
            entry('_u', struct=self.model._u),
            entry('_p', struct=self.model._p),
            entry('_tvp', struct=self.model._tvp),
            entry('_w', struct=self.model._w)
        ])

        # Initiate numerical structures to store the solutions (updated at each iteration)
        self.sim_x_num = self.sim_x(0)
        self.sim_z_num = self.sim_z(0)
        self.sim_p_num = self.sim_p(0)
        self.sim_aux_num = self.model._aux_expression(0)

        if self.model.model_type == 'discrete':

            # Build the rhs expression with the newly created variables
            alg = self.model._alg_fun(sim_x['_x'],sim_p['_u'],sim_z['_z'],sim_p['_tvp'],sim_p['_p'], sim_p['_w'])
            x_next = self.model._rhs_fun(sim_x['_x'],sim_p['_u'],sim_z['_z'],sim_p['_tvp'],sim_p['_p'], sim_p['_w'])

            # Build the DAE function
            nlp = {'x': sim_z['_z'], 'p': vertcat(sim_x['_x'], sim_p), 'f': DM(0), 'g': alg}
            self.discrete_dae_solver = nlpsol('dae_roots', 'ipopt', nlp)

            # Build the simulator function:
            self.simulator = Function('simulator',[sim_x['_x'], sim_z['_z'], sim_p],[x_next])


        elif self.model.model_type == 'continuous':

            # Define the ODE
            xdot = self.model._rhs_fun(sim_x['_x'],sim_p['_u'],sim_z['_z'],sim_p['_tvp'],sim_p['_p'], sim_p['_w'])
            alg = self.model._alg_fun(sim_x['_x'],sim_p['_u'],sim_z['_z'],sim_p['_tvp'],sim_p['_p'], sim_p['_w'])

            dae = {
                'x': sim_x,
                'z': sim_z,
                'p': sim_p,
                'ode': xdot,
                'alg': alg,
            }

            # Set the integrator options
            opts = {
                'abstol': self.abstol,
                'reltol': self.reltol,
                'tf': self.t_step
            }

            # Build the simulator
            self.simulator = integrator('simulator', self.integration_tool, dae,  opts)

        sim_aux = self.model._aux_expression_fun(sim_x['_x'],sim_p['_u'],sim_z['_z'],sim_p['_tvp'],sim_p['_p'])
        # Create function to caculate all auxiliary expressions:
        self.sim_aux_expression_fun = Function('sim_aux_expression_fun', [sim_x, sim_z, sim_p], [sim_aux])

        self.flags['setup'] = True


    def set_param(self, **kwargs):
        """Set the parameters for the simulator. Setting the simulation time step t_step is necessary for setting up the simulator via setup_simulator.

        :param integration_tool: Sets which integration tool is used, defaults to ``cvodes`` (only continuous)
        :type integration_tool: string
        :param abstol: gives the maximum allowed absolute tolerance for the integration, defaults to ``1e-10`` (only continuous)
        :type abstol: float
        :param reltol: gives the maximum allowed relative tolerance for the integration, defaults to ``1e-10`` (only continuous)
        :type abstol: float
        :param t_step: Sets the time step for the simulation
        :type t_step: float

        :return: None
        :rtype: None
        """
        assert self.flags['setup'] == False, 'Cannot call set_param after simulator was setup.'

        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for {} model.'.format(key, self.model.model_type))
            setattr(self, key, value)


    def get_tvp_template(self):
        """Obtain the output template for :py:func:`set_tvp_fun`.
        Use this method in conjunction with :py:func:`set_tvp_fun`
        to define the function for retrieving the time-varying parameters at each sampling time.

        :return: numerical CasADi structure
        :rtype: struct_SX
        """
        return self.model._tvp(0)


    def set_tvp_fun(self,tvp_fun):
        """Method to set the function which returns the values of the time-varying parameters.
        This function must return a CasADi structure which can be obtained with :py:func:`get_tvp_template`.

        In the :py:class:`do_mpc.model.Model` we have defined the following parameters:

        ::

            a = model.set_variable('_tvp', 'a')

        The integrate the ODE or evaluate the discrete dynamics, the simulator needs
        to obtain the numerical values of these parameters at each timestep.
        In the most general case, these values can change,
        which is why a function must be supplied that can be evaluted at each timestep to obtain the current values.

        **do-mpc** requires this function to have a specific return structure which we obtain first by calling:

        ::

            tvp_template = simulator.get_tvp_template()

        The time-varying parameter function can look something like this:

        ::

            def tvp_fun(t_now):
                tvp_template['a'] = 3
                return tvp_template

            simulator.set_tvp_fun(tvp_fun)

        which results in constant parameters.

        .. note::

            From the perspective of the simulator there is no difference between
            time-varying parameters and regular parameters. The difference is important only
            for the MPC controller and MHE estimator. These methods consider a finite sequence
            of future / past information, e.g. the weather, which can change over time.
            Parameters, on the other hand, are constant over the entire horizon.

        :param tvp_fun: Function which gives the values of the time-varying parameters
        :type tvp_fun: function

        :raises assertion: tvp_fun has incorrect return type.
        :raises assertion: Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.

        :return: None
        :rtype: None
        """
        assert isinstance(tvp_fun(0), structure3.DMStruct), 'tvp_fun has incorrect return type.'
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        self.tvp_fun = tvp_fun

        self.flags['set_tvp_fun'] = True


    def get_p_template(self):
        """Obtain output template for :py:func:`set_p_fun`.
        Use this method in conjunction with :py:func:`set_p_fun`
        to define the function for retrieving the parameters at each sampling time.

        See :py:func:`set_p_fun` for more details.

        :return: numerical CasADi structure
        :rtype: struct_SX
        """
        return self.model._p(0)


    def set_p_fun(self,p_fun):
        """Method to set the function which gives the values of the parameters.
        This function must return a CasADi structure which can be obtained with :py:func:`get_p_template`.

        **Example**:

        In the :py:class:`do_mpc.model.Model` we have defined the following parameters:

        ::

            Theta_1 = model.set_variable('parameter', 'Theta_1')
            Theta_2 = model.set_variable('parameter', 'Theta_2')
            Theta_3 = model.set_variable('parameter', 'Theta_3')

        To integrate the ODE or evaluate the discrete dynamics, the simulator needs
        to obtain the numerical values of these parameters at each timestep.
        In the most general case, these values can change,
        which is why a function must be supplied that can be evaluted at each timestep to obtain the current values.

        **do-mpc** requires this function to have a specific return structure which we obtain first by calling:

        ::

            p_template = simulator.get_p_template()

        The parameter function can look something like this:

        ::

            p_template['Theta_1'] = 2.25e-4
            p_template['Theta_2'] = 2.25e-4
            p_template['Theta_3'] = 2.25e-4

            def p_fun(t_now):
                return p_template

            simulator.set_p_fun(p_fun)

        which results in constant parameters.

        A more "interesting" variant could be this random-walk:

        ::

            p_template['Theta_1'] = 2.25e-4
            p_template['Theta_2'] = 2.25e-4
            p_template['Theta_3'] = 2.25e-4

            def p_fun(t_now):
                p_template['Theta_1'] += 1e-6*np.random.randn()
                p_template['Theta_2'] += 1e-6*np.random.randn()
                p_template['Theta_3'] += 1e-6*np.random.randn()
                return p_template



        :param p_fun: A function which gives the values of the parameters
        :type p_fun: python function

        :raises assert: p must have the right structure

        :return: None
        :rtype: None
        """
        assert isinstance(p_fun(0), structure3.DMStruct), 'p_fun has incorrect return type.'
        assert self.get_p_template().labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun
        self.flags['set_p_fun'] = True


    def set_initial_guess(self):
        """Initial guess for DAE variables.
        Use the current class attribute :py:attr:`z0` to create the initial guess for the DAE algebraic equations.

        The simulator uses "warmstarting" to solve the continous/discrete DAE system by using the previously computed
        algebraic states as an initial guess. Thus, this method is typically only invoked once.

        .. warning::
            If no initial values for :py:attr:`z0` were supplied during setup, they default to zero.

        """
        assert self.flags['setup'] == True, 'MPC was not setup yet. Please call MPC.setup().'

        self.sim_z_num['_z'] = self._z0.cat

    def simulate(self):
        """Call the CasADi simulator.

        .. warning::

            :py:func:`simulate` can be used as part of the public API but is typically
            called from within :py:func:`make_step` which wraps this method and sets the
            required values to the ``sim_x_num`` and ``sim_p_num`` structures automatically.

        Numerical values for ``sim_x_num`` and ``sim_p_num`` need to be provided beforehand
        in order to simulate the system for one time step:

        * states ``sim_c_num['_x']``

        * algebraic states ``sim_z_num['_z']``

        * inputs ``sim_p_num['_u']``

        * parameter ``sim_p_num['_p']``

        * time-varying parameters ``sim_p_num['_tvp']``

        The function returns the new state of the system.

        :return: x_new
        :rtype: numpy array
        """
        assert self.flags['setup'] == True, 'Simulator is not setup. Call simulator.setup() first.'

        # extract numerical values
        sim_x_num = self.sim_x_num
        sim_z_num = self.sim_z_num
        sim_p_num = self.sim_p_num

        if self.model.model_type == 'discrete':
            if self.model.n_z > 0: # Solve DAE only when it exists ...
                r = self.discrete_dae_solver(x0 = sim_z_num, ubg = 0, lbg = 0, p=vertcat(sim_x_num,sim_p_num))
                sim_z_num.master = r['x']
            x_new = self.simulator(sim_x_num, sim_z_num, sim_p_num)
        elif self.model.model_type == 'continuous':
            r = self.simulator(x0 = sim_x_num, z0 = sim_z_num, p = sim_p_num)
            x_new = r['xf']
            z_now = r['zf']
            sim_z_num.master = z_now

        aux_now = self.sim_aux_expression_fun(sim_x_num, sim_z_num, sim_p_num)

        self.sim_aux_num.master = aux_now

        return x_new

    def make_step(self, u0, v0=None, w0=None):
        """Main method of the simulator class during control runtime. This method is called at each timestep
        and computes the next state or the current control input :py:obj:`u0`. The method returns the resulting measurement,
        as defined in :py:class:`do_mpc.model.Model.set_meas`.

        The initial state :py:attr:`x0` is stored as a class attribute. Use this attribute :py:attr:`x0` to change the initial state.
        It is also possible to supply an initial guess for the algebraic states through the attribute :py:attr:`z0` and by calling
        :py:func:`set_initial_guess`.

        Finally, the method can be called with values for the process noise ``w0`` and the measurement noise ``v0``
        that were (optionally) defined in the :py:class:`do_mpc.model.Model`.
        Typically, these values should be sampled from a random distribution, e.g. ``np.random.randn`` for a random normal distribution.

        The method prepares the simulator by setting the current parameters, calls :py:func:`simulator.simulate`
        and updates the :py:class:`do_mpc.data` object.

        :param u0: Current input to the system.
        :type u0: numpy.ndarray

        :param v0: Additive measurement noise
        :type v0: numpy.ndarray (optional)

        :param w0: Additive process noise
        :type w0: numpy.ndarray (optional)

        :return: y_next
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Simulator is not setup. Call simulator.setup() first.'
        assert isinstance(u0, (np.ndarray, casadi.DM, structure3.DMStruct)), 'u0 is wrong input type. You have: {}'.format(type(u0))
        assert u0.shape == self.model._u.shape, 'u0 has incorrect shape. You have: {}, expected: {}'.format(u0.shape, self.model._u.shape)
        assert isinstance(u0, (np.ndarray, casadi.DM, structure3.DMStruct)), 'u0 is wrong input type. You have: {}'.format(type(u0))
        assert u0.shape == self.model._u.shape, 'u0 has incorrect shape. You have: {}, expected: {}'.format(u0.shape, self.model._u.shape)

        if w0 is None:
            w0 = self.model._w(0)
        else:
            input_types = (np.ndarray, casadi.DM, structure3.DMStruct)
            assert isinstance(w0, input_types), 'w0 is wrong input type. You have: {}. Must be of type'.format(type(w0), input_types)
            assert w0.shape == self.model._w.shape, 'w0 has incorrect shape. You have: {}, expected: {}'.format(w0.shape, self.model._w.shape)

        if v0 is None:
            v0 = self.model._v(0)
        else:
            input_types = (np.ndarray, casadi.DM, structure3.DMStruct)
            assert isinstance(v0, input_types), 'v0 is wrong input type. You have: {}. Must be of type'.format(type(v0), input_types)
            assert v0.shape == self.model._v.shape, 'v0 has incorrect shape. You have: {}, expected: {}'.format(v0.shape, self.model._v.shape)

        tvp0 = self.tvp_fun(self._t0)
        p0 = self.p_fun(self._t0)
        t0 = self._t0
        x0 = self._x0
        self.sim_x_num['_x'] = x0
        self.sim_p_num['_u'] = u0
        self.sim_p_num['_p'] = p0
        self.sim_p_num['_tvp'] = tvp0
        self.sim_p_num['_w'] = w0

        x_next = self.simulate()

        z0 = self.sim_z_num['_z']
        aux0 = self.sim_aux_num

        # Call measurement function
        y_next = self.model._meas_fun(x_next, u0, z0, tvp0, p0, v0)

        self.data.update(_x = x0)
        self.data.update(_u = u0)
        self.data.update(_z = z0)
        self.data.update(_tvp = tvp0)
        self.data.update(_p = p0)
        self.data.update(_y = y_next)
        self.data.update(_aux = aux0)
        self.data.update(_time = t0)

        self._x0.master = x_next
        self._z0.master = z0
        self._u0.master = u0
        self._t0 = self._t0 + self.t_step

        return y_next.full()
