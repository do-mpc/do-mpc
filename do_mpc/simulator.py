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


"""
Simulate continous-time ODE/DAE or discrete-time dynamic systems.
"""

import numpy as np
import casadi.tools as castools
import pdb
import warnings
import do_mpc
from typing import Union,Callable
from dataclasses import dataclass
from typing import Dict


# Define what is included in the Sphinx documentation.
__all__ = ['Simulator', 'SimulatorSettings', 'ContinousSimulatorSettings']

@dataclass
class SimulatorSettings:
    """Settings for :py:class:`Simulator`.
    An instance of this class is automatically generated as the attribute ``settings`` when creating the :py:class:`Simulator`.

    **Example**:

    ::

        simulator = do_mpc.simulator.Simulator(model)
        simulator.settings.t_step = 0.5

    """
    t_step: float = None
    """Timestep of the Simulator"""
    
    def check_for_mandatory_settings(self):
        """Method to assert the necessary settings required to design :py:class:`do_mpc.controller`
        """
        if self.t_step is None:
            raise ValueError("t_step must be set")


class ContinousSimulatorSettings(SimulatorSettings):
    """Settings for :py:class:`Simulator` for continous-time systems.
    

    An instance of this class is automatically generated as the attribute ``settings`` when creating the :py:class:`Simulator`.

    **Example**:

    ::

        simulator = do_mpc.simulator.Simulator(model)
        simulator.settings.t_step = 0.5
    

    Note:     
            As version 4.6.3, additional CasADI integrator options can be accessed as can be seen in the example below:

    **Example**:

    ::

        simulator = do_mpc.simulator.Simulator(model)
        simulator.settings.integration_opts = {'gather_stats':True, 'print_stats': True, 'verbose':False}

    """
    
    abstol: float = 1e-10
    """Absolute tolerance for the integrator"""

    reltol: float = 1e-10
    """Relative tolerance for the integrator"""

    integration_tool: str = 'cvodes'
    """Integration tool to be used. Options are 'cvodes' and 'idas'"""

    integration_opts: Dict = {}
    """Dictionary with options for the CasADi integrator call. Used tu update the opts dict in :py:func:`setup`.
    
    All options are listed `here <https://casadi.sourceforge.net/api/html/db/d3d/classcasadi_1_1Integrator.html>`_."""



class Simulator(do_mpc.model.IteratedVariables):
    """A class for simulating systems. Discrete-time and continuous systems can be considered.
    
    .. versionadded:: >v4.5.1

        New interface to settings. The class has an attribute ``settings`` which is an instance of :py:class:`SimulatorSettings` or :py:class:`ContinousSimulatorSettings`
        (please see this documentation for a list of available settings).
        Settings are now chosen as:

        ::

            simulator.settings.t_step = 0.5
        
        Previously, settings were passed to :py:meth:`set_param`. This method is still available and wraps the new interface.
        The new method has important advantages:
        
        1. The ``simulator.settings`` attribute can be printed to see the current configuration.
        
        2. Context help is available in most IDEs (e.g. VS Code) to see the available settings, the type and a description.


    **do-mpc** uses the CasADi interface to popular state-of-the-art tools such as Sundials `CVODES`_
    for the integration of ODE/DAE equations.

    .. _CVODES: https://computing.llnl.gov/projects/sundials

    **Configuration and setup:**

    Configuring and setting up the simulator involves the following steps:

    1. Configure the simulator with :py:class:`SimulatorSettings` or :py:class:`ContinousSimulatorSettings`. The simulator instance has the attribute ``settings`` which is an instance of :py:class:`SimulatorSettings` or :py:class:`ContinousSimulatorSettings`.

    2. Set parameter function with :py:func:`get_p_template` and  :py:func:`set_p_fun`.

    3. Set time-varying parameter function with :py:func:`get_tvp_template` and  :py:func:`set_tvp_fun`.

    4. Setup simulator with :py:func:`setup`.

    During runtime, call the simulator :py:func:`make_step` method with current input (``u``).
    This computes the next state of the system and the respective measurement.
    Optionally, pass (sampled) random variables for the process ``w`` and measurement noise ``v`` (if they were defined in :py:class`do_mpc.model.Model`)

    Args:
        model: A configured and setup :py:class:`do_mpc.model.Model`
    """
    def __init__(self, model:do_mpc.model.Model):

        self.model = model
        do_mpc.model.IteratedVariables.__init__(self)

        assert model.flags['setup'] == True, 'Model for simulator was not setup. After the complete model creation call model.setup().'

        self.data = do_mpc.data.Data(model)

        if self.model.model_type == 'continuous':
            self._settings = ContinousSimulatorSettings()
        elif self.model.model_type == 'discrete':
            self._settings = SimulatorSettings()

        self.flags = {
            'set_tvp_fun': False,
            'set_p_fun': False,
            'setup': False,
            'first_step': True,
        }

        # Initialze the private attributes for scaling of variables.
        # We only scale differential and algebraic variables because control actions and parameters are constant during integration. 
        self._x_scaling = self.model._x(1.0)
        self._z_scaling = self.model._z(1.0)

    @property
    def settings(self):
        '''
        All necessary parameters for the simulator.

        This is a core attribute of the Simulator class. It is used to set and change parameters when setting up the simulator
        by accessing an instance of :py:class:`SimulatorSettings` or :py:class:`ContinousSimulatorSettings`. 
        
        Example to change settings:

        ::

            simulator.settings.t_step = 0.5

        Note:     
            Settings cannot be updated after calling :py:meth:`do_mpc.simulator.setup`.

        For a detailed list of all available parameters see :py:class:`SimulatorSettings` or :py:class:`ContinousSimulatorSettings`. 
        '''
        return self._settings
    
    @settings.setter
    def settings(self, val):
        warnings.warn('Cannot change the settings attribute')

    @do_mpc.tools.IndexedProperty
    def scaling(self, ind):
        """Get or set scaling factors for differential and algebraic states.
        
        Scaling can significantly improve the numerical stability of the simulator,
        especially when differential and algebraic states have very different orders of magnitude.
        Variables are internally scaled (divided by the scaling factor) before integration,
        then scaled back to their physical values after integration. The scaled system is integrated.
        
        Query and set scaling of the state variables.
        The :py:func:`Simulator.scaling` method is an indexed property, meaning
        getting and setting this property requires an index and calls this function.
        The power index (elements are seperated by comas) must contain atleast the following elements:

        ======      =================   ==========================================================
        order       index name          valid options
        ======      =================   ==========================================================
        1           variable type       ``_x`` and ``_z``
        2           variable name       Names defined in :py:class:`do_mpc.model.Model`.
        ======      =================   ==========================================================
        
        **Example:**
        
        ::
        
            # Set scaling for state variable 'x3' to 100
            simulator.scaling['_x', 'x3'] = 100
            
            # Set scaling for algebraic variable 'z1' to 0.001  
            simulator.scaling['_z', 'z1'] = 0.001
            
            # Get current scaling value
            x3_scaling = simulator.scaling['_x', 'x3']
        
        **When to use scaling:**
        
        - When state variables differ by several orders of magnitude
        (e.g., temperatures ~300K and concentrations ~1e-6 mol/L)
        - When experiencing numerical difficulties

            
        Note:
            Scaling factors must be set before calling simulator.setup().
            :py:meth:`setup`. Default scaling is 1.0 for all variables.
        """

        assert isinstance(ind, tuple), 'Power index must include var_type, var_name (as a tuple).'
        assert len(ind)>=2, 'Power index must include var_type, var_name (as a tuple).'

        var_type   = ind[0]
        var_name   = ind[1:]

        err_msg = 'Invalid power index {} for var_type. Must be from (_x, states, _z, algebraic).'
        assert var_type in ('_x', '_z'), err_msg.format(var_type)

        query = '{var_type}_scaling'.format(var_type=var_type)
        # query results string e.g. _x_scaling, _u_scaling

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .scaling with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

        return var_struct[var_name]

    @scaling.setter
    def scaling(self, ind, val):
        """Set scaling factor for a specific variable.
        """
        assert not self.flags['setup'], 'Scaling can only be set before the simulator is set up.'
        assert isinstance(ind, tuple), 'Power index must include var_type, var_name (as a tuple).'
        assert len(ind)>=2, 'Power index must include var_type, var_name (as a tuple).'
        var_type   = ind[0]
        var_name   = ind[1:]

        err_msg = 'Invalid power index {} for var_type. Must be from (_x, states, _z, algebraic).'
        assert var_type in ('_x', '_z'), err_msg.format(var_type)

        query = '{var_type}_scaling'.format(var_type=var_type)
        # query results string e.g. _x_scaling, _u_scaling

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .scaling with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

        var_struct[var_name] = val
    
    def reset_history(self)->None:
        """Reset the history of the simulator.
        """
        self._t0 = np.array([0])
        self.data.init_storage()
        self.flags['first_step'] = True

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

        self._settings.check_for_mandatory_settings()


    def setup(self)->None:
        """Sets up the simulator and finalizes the simulator configuration.
        Only after the setup, the :py:func:`make_step` method becomes available.

        Raises:
            assertion: t_step must be set
        """

        self._check_validity()

        self.sim_x = sim_x =  self.model.sv.sym_struct([
            castools.entry('_x', struct=self.model._x)
            ])
        self.sim_z = sim_z =  self.model.sv.sym_struct([
            castools.entry('_z', struct=self.model._z)
            ])

        self.sim_p = sim_p = self.model.sv.sym_struct([
            castools.entry('_u', struct=self.model._u),
            castools.entry('_p', struct=self.model._p),
            castools.entry('_tvp', struct=self.model._tvp),
            castools.entry('_w', struct=self.model._w)
        ])

        # Create scaling struct and assign values for _x and _z
        self.sim_x_scaling = sim_x_scaling = sim_x(1.0)
        self.sim_z_scaling = sim_z_scaling = sim_z(1.0)
        sim_x_scaling["_x"] = self._x_scaling
        sim_z_scaling["_z"] = self._z_scaling

        # Create the unscaled (physical) variables
        self.sim_x_unscaled = sim_x_unscaled = sim_x(sim_x.cat * sim_x_scaling)
        self.sim_z_unscaled = sim_z_unscaled = sim_z(sim_z.cat * sim_z_scaling)

        # Initiate numerical structures to store the solutions (updated at each iteration)
        self.sim_x_num = self.sim_x(0)
        self.sim_x_num_unscaled = self.sim_x(0)
        self.sim_z_num = self.sim_z(0)
        self.sim_z_num_unscaled = self.sim_z(0)
        self.sim_p_num = self.sim_p(0)
        self.sim_aux_num = self.model._aux_expression(0)

        if self.model.model_type == 'discrete':

            # Build the rhs expression with the newly created variables
            # NOTE: _alg_fun is evaluated with the unscaled variables to introduce the scaling factors. 
            #       During evaluation the scaled variables can then be used.
            alg = self.model._alg_fun(sim_x_unscaled['_x'], sim_p['_u'], sim_z_unscaled['_z'], sim_p['_tvp'], sim_p['_p'], sim_p['_w'])
            
            # Do the same for the ode expression but also divide by the scaling factor of the states.
            x_next = self.model._rhs_fun(sim_x_unscaled['_x'], sim_p['_u'], sim_z_unscaled['_z'], sim_p['_tvp'], sim_p['_p'], sim_p['_w']) / sim_x_scaling

            # Build the DAE function
            nlp = {'x': sim_z['_z'], 'p': castools.vertcat(sim_x['_x'], sim_p), 'f': castools.DM(0), 'g': alg}
            self.discrete_dae_solver = castools.nlpsol('dae_roots', 'ipopt', nlp)

            # Build the simulator function:
            self.simulator = castools.Function('simulator',[sim_x['_x'], sim_z['_z'], sim_p],[x_next])


        elif self.model.model_type == 'continuous':

            # Define the ODE
            # NOTE: We evaluate here with the unscaled variables to introduce the scaling factors in the equations.
            # We have to divide the dynamics by the scaling factor of the states to get the correct result.
            # From now on we can use the scaled variables.
            xdot = self.model._rhs_fun(sim_x_unscaled['_x'], sim_p['_u'], sim_z_unscaled['_z'], sim_p['_tvp'], sim_p['_p'], sim_p['_w']) / self._x_scaling
            alg = self.model._alg_fun(sim_x_unscaled['_x'], sim_p['_u'], sim_z_unscaled['_z'], sim_p['_tvp'], sim_p['_p'], sim_p['_w'])

            # Now setup the dae system with the scaled variables
            self.dae = dae = {
                'x': sim_x,
                'z': sim_z,
                'p': sim_p,
                'ode': xdot,
                'alg': alg,
            }

            opts = {}
            # Set the integrator options, note that 'abstol' and 'reltol' are not needed for collocation
            if self._settings.integration_tool != 'collocation':
                opts = {
                    'abstol': self._settings.abstol,
                    'reltol': self._settings.reltol,
                }

            # Add further options for the CasADi integrator call defined by the user 
            opts.update(self._settings.integration_opts)

            if do_mpc.CASADI_LEGACY_MODE:
                opts['tf'] = self._settings.t_step
                self.simulator = castools.integrator('simulator', self._settings.integration_tool, dae, opts)
            else:
                # Build the simulator
                t0 = 0.0
                self.simulator = castools.integrator('simulator', self._settings.integration_tool, dae, t0, self._settings.t_step, opts)

        # Evaluate symbolically with unscaled variables such that the scaled variables can be used during evaluation.
        sim_aux = self.model._aux_expression_fun(sim_x_unscaled['_x'], sim_p['_u'], sim_z_unscaled['_z'], sim_p['_tvp'], sim_p['_p'])
        # Create function to caculate all auxiliary expressions:
        self.sim_aux_expression_fun = castools.Function('sim_aux_expression_fun', [sim_x, sim_z, sim_p], [sim_aux])

        self.flags['setup'] = True


    def set_param(self, **kwargs)->None:
        """
        Warnings:
            This method will be depreciated in a future version. Settings are available via the :py:attr:`settings` attribute which is an instance of :py:class:`ContinousSimulatorSettings` or :py:class:`SimulatorSettings`.

        Note:
            A comprehensive list of all available parameters can be found in :py:class:`ContinousSimulatorSettings` or :py:class:`SimulatorSettings`.
        
        For example:
        
        ::

            simulator.settings.t_step = 0.5
        
        The old interface, as shown in the example below, can still be accessed until further notice.
        
        ::

            simulator.set_param(t_step=0.5)

        Note: 
            The only required parameters  are ``t_step``. All other parameters are optional.

        """
        assert self.flags['setup'] == False, 'Setting parameters after setup is prohibited.'

        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                    setattr(self._settings, key, value)
            else:
                print('Warning: Key {} does not exist for Simulator.'.format(key))


    def get_tvp_template(self)->Union[castools.structure3.SXStruct,castools.structure3.MXStruct]:
        """Obtain the output template for :py:func:`set_tvp_fun`.
        Use this method in conjunction with :py:func:`set_tvp_fun`
        to define the function for retrieving the time-varying parameters at each sampling time.

        Returns:
            numerical CasADi structure
        """
        return self.model._tvp(0)


    def set_tvp_fun(self,tvp_fun:Callable[[float],Union[castools.structure3.SXStruct,castools.structure3.MXStruct]])->None:
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

        Note:
            From the perspective of the simulator there is no difference between
            time-varying parameters and regular parameters. The difference is important only
            for the MPC controller and MHE estimator. These methods consider a finite sequence
            of future / past information, e.g. the weather, which can change over time.
            Parameters, on the other hand, are constant over the entire horizon.

        Args:
            tvp_fun: Function which gives the values of the time-varying parameters

        Raises:
            assertion: tvp_fun has incorrect return type.
            assertion: Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.
        """
        assert isinstance(tvp_fun(0), castools.structure3.DMStruct), 'tvp_fun has incorrect return type.'
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        self.tvp_fun = tvp_fun

        self.flags['set_tvp_fun'] = True


    def get_p_template(self)->Union[castools.structure3.SXStruct,castools.structure3.MXStruct]:
        """Obtain output template for :py:func:`set_p_fun`.
        Use this method in conjunction with :py:func:`set_p_fun`
        to define the function for retrieving the parameters at each sampling time.

        See :py:func:`set_p_fun` for more details.

        Returns:
            numerical CasADi structure
        """
        return self.model._p(0)


    def set_p_fun(self,p_fun:Callable[[float],Union[castools.structure3.SXStruct,castools.structure3.MXStruct]])->None:
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

        Args:
            p_fun: A function which gives the values of the parameters

        Raises:
            assert: p must have the right structure
        """
        assert isinstance(p_fun(0), castools.structure3.DMStruct), 'p_fun has incorrect return type.'
        assert self.get_p_template().labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun
        self.flags['set_p_fun'] = True


    def set_initial_guess(self)->None:
        """Initial guess for DAE variables.
        Use the current class attribute :py:attr:`z0` to create the initial guess for the DAE algebraic equations.

        The simulator uses "warmstarting" to solve the continous/discrete DAE system by using the previously computed
        algebraic states as an initial guess. Thus, this method is typically only invoked once.

        Warnings:
            If no initial values for :py:attr:`z0` were supplied during setup, they default to zero.
        """
        assert self.flags['setup'] == True, 'Simulator was not setup yet. Please call Simulator.setup().'

        # We assume that the unscaled z0 is provided by the user. Hence, we have to scale it before we can use it in the DAE solver.
        self.sim_x_num["_x"] = self._x0.cat / self._x_scaling
        self.sim_x_num_unscaled["_x"] = self._x0.cat
        self.sim_z_num['_z'] = self._z0.cat / self._z_scaling
        self.sim_z_num_unscaled['_z'] = self._z0.cat

    def init_algebraic_variables(self) -> np.ndarray:
        """Initializes the algebraic variables. 
        Solve the algebraic equations for the initial values of :py:attr:`x0`, :py:attr:`u0`, :py:attr:`p0`, :py:attr:`tvp0`.
        Sets the results to :py:attr:`z0` and returns them. 

        Note:
            The method internally calls :py:func:`set_initial_guess` to set the initial guess for the algebraic variables.

        The initialization is computed by solving the algebraic model equations under consideration of the initial guess supplied in :py:attr:`z0`.

        **Example**:

        ::

            simulator = do_mpc.simulator.Simulator(model)

            # Set initial value for the state:
            simulator.x0 = np.array([0.1, 0.1]).reshape(-1,1)

            # Obtain initial guess for the algebraic variables:
            z0 = simulator.init_algebraic_variables()

            # Initial guess is stored in simulator.z0 and simulator.set_initial_guess() was called internally.


        Returns:
            Initial guess for the algebraic variables.

        """
        if self.model.flags['setup'] is False:
            raise RuntimeError(
                'The model must be setup before the algebraic variables can be initialized.'
            )


        z0 = castools.vertcat(self.z0) / self._z_scaling
        x0 = castools.vertcat(self.x0) / self._x_scaling
        p0 = castools.vertcat(x0, self.u0, self.p_fun(self.t0), self.tvp_fun(self.t0))

        if self.model.model_type == 'discrete':
            res = self.discrete_dae_solver(x0 = z0, ubg = 0, lbg = 0, p=p0)

        elif self.model.model_type == 'continuous':
            residual_to_initial_guess = castools.vertcat(self.sim_z["_z"]) - z0
            cost = castools.sum2(castools.sum1(residual_to_initial_guess**2))

            nlp = {}
            nlp['x'] = castools.vertcat(self.sim_z["_z"])
            nlp['f'] = cost
            nlp['g'] = castools.vertcat(self.dae["alg"])
            nlp['p'] = castools.vertcat(self.sim_x["_x"], self.sim_p)

            suppress_ipopt =  {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}

            solver = castools.nlpsol("solver", "ipopt", nlp, suppress_ipopt)
            res = solver(x0=z0, lbg=0, ubg=0, p=p0)
        else:
            raise ValueError(f'Model type {self.model.model_type} is not supported.')
        

        z_init = res['x']

        self.z0 = z_init * self._z_scaling

        self.set_initial_guess()

        return self.z0


    def simulate(self)->np.ndarray:
        """Call the CasADi simulator.

        Warnings:
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

        Returns:
            x_new
        """
        assert self.flags['setup'] == True, 'Simulator is not setup. Call simulator.setup() first.'

        # extract numerical values
        sim_x_num = self.sim_x_num
        sim_z_num = self.sim_z_num
        sim_p_num = self.sim_p_num

        if self.model.model_type == 'discrete':
            if self.model.n_z > 0: # Solve DAE when it exists...
                r = self.discrete_dae_solver(x0 = sim_z_num, ubg = 0, lbg = 0, p=castools.vertcat(sim_x_num, sim_p_num))
                sim_z_num["_z"] = r['x']
            x_new = self.simulator(sim_x_num, sim_z_num, sim_p_num)
            # NOTE: This z_new actually satisfies the AE before the integration takes place, so g(x_k, u_k, z_new, p_k) = 0.
            # If you would like to have the z_new satisfying the AE after the integration, you need to call the DAE solver again, so g(x_new, u_k, z_new, p_k) = 0.
            # Further, be careful because while x_new remains the starting point for the next integration, u_{k+1} will change and hence also z_new, so g(x_new, u_{k+1}, z_new, p_{k+1}) = 0.
            z_new = sim_z_num.cat

        elif self.model.model_type == 'continuous':
            r = self.simulator(x0 = sim_x_num, z0 = sim_z_num, p = sim_p_num)
            x_new = r['xf']
            z_new = r['zf']
        else:
            raise ValueError(f'Model type {self.model.model_type} is not supported.')
        
        # Update all numerical values in the sim_x_num and sim_z_num structures
        self.sim_x_num.master = x_new
        self.sim_x_num_unscaled.master = x_new * self._x_scaling.cat
        self.sim_z_num.master = z_new
        self.sim_z_num_unscaled.master = z_new * self._z_scaling.cat
        # if z_new.shape[0] > 0:
        #     self.sim_z_num_unscaled.master = z_new * self._z_scaling.cat

        # There may be made an error here. sim_p_num fits to values in time step
        # k + 1 (new). However, the values are actually the p values for step
        # k (now).
        aux_new = self.sim_aux_expression_fun(self.sim_x_num, self.sim_z_num, sim_p_num)

        self.sim_aux_num.master = aux_new

        return x_new, z_new

    def make_step(self, u0:np.ndarray=None, v0:np.ndarray=None, w0:np.ndarray=None)-> np.ndarray:
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

        Args:
            u0: Current input to the system. Optional parameter for autonomous systems.
            v0: Additive measurement noise
            w0: Additive process noise
        
        Returns:
            y_next
        """
        # Generate dummy input if system is autnomous

        if u0 is None:
            assert self.model.n_u == 0, 'No input u0 provided. Please provide an input u0.'
            u0 = self.model._u(0)

        assert self.flags['setup'] == True, 'Simulator is not setup. Call simulator.setup() first.'
        assert isinstance(u0, (np.ndarray, castools.DM, castools.structure3.DMStruct)), 'u0 is wrong input type. You have: {}'.format(type(u0))
        assert u0.shape == self.model._u.shape, 'u0 has incorrect shape. You have: {}, expected: {}'.format(u0.shape, self.model._u.shape)
        assert isinstance(u0, (np.ndarray, castools.DM, castools.structure3.DMStruct)), 'u0 is wrong input type. You have: {}'.format(type(u0))
        assert u0.shape == self.model._u.shape, 'u0 has incorrect shape. You have: {}, expected: {}'.format(u0.shape, self.model._u.shape)

        if w0 is None:
            w0 = self.model._w(0)
        else:
            input_types = (np.ndarray, castools.DM, castools.structure3.DMStruct)
            assert isinstance(w0, input_types), 'w0 is wrong input type. You have: {}. Must be of type'.format(type(w0), input_types)
            assert w0.shape == self.model._w.shape, 'w0 has incorrect shape. You have: {}, expected: {}'.format(w0.shape, self.model._w.shape)

        if v0 is None:
            v0 = self.model._v(0)
        else:
            input_types = (np.ndarray, castools.DM, castools.structure3.DMStruct)
            assert isinstance(v0, input_types), 'v0 is wrong input type. You have: {}. Must be of type'.format(type(v0), input_types)
            assert v0.shape == self.model._v.shape, 'v0 has incorrect shape. You have: {}, expected: {}'.format(v0.shape, self.model._v.shape)

        tvp0 = self.tvp_fun(self._t0)
        p0 = self.p_fun(self._t0)
        t0 = self._t0
        x0 = self._x0

        z0 = self.sim_z_num['_z']
        z0_unscaled = self.sim_z_num_unscaled["_z"]
        self.sim_x_num['_x'] = x0.cat / self._x_scaling
        self.sim_p_num['_u'] = u0
        self.sim_p_num['_p'] = p0
        self.sim_p_num['_tvp'] = tvp0
        self.sim_p_num['_w'] = w0

        if self.flags['first_step']:
            # Remember to plug in the unscaled (physical) version of x and z
            aux0 = self.sim_aux_expression_fun(self.sim_x_num["_x"], self.sim_z_num["_z"], self.sim_p_num)
        else:
            # .master is chosen so that a copy is created of the variables.
            aux0 = self.sim_aux_num.master

        x_next, z_next = self.simulate()

        # Call measurement function
        x_next_unscaled = x_next * self._x_scaling.cat
        z_next_unscaled = z_next * self._z_scaling.cat

        y_next = self.model._meas_fun(x_next_unscaled, u0, z_next_unscaled, tvp0, p0, v0)

        # Update data object
        self.data.update(_x = x0.cat)
        self.data.update(_u = u0)
        self.data.update(_z = z0_unscaled)
        self.data.update(_tvp = tvp0)
        self.data.update(_p = p0)
        self.data.update(_y = y_next)
        self.data.update(_aux = aux0)
        self.data.update(_time = t0)

        self._x0.master = x_next_unscaled
        self._z0.master = z_next_unscaled
        self._u0.master = u0
        self._t0 = self._t0 + self._settings.t_step 

        self.flags['first_step'] = False

        return y_next.full()

