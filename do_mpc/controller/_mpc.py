
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
import casadi.tools as castools
import pdb
import itertools
import time
import warnings
from do_mpc.tools import IndexedProperty
from do_mpc.tools._casstructure import _SymVar
from typing import Union, Callable, Optional
from dataclasses import asdict
import do_mpc
from ._controllersettings import MPCSettings

class MPC(do_mpc.optimizer.Optimizer, do_mpc.model.IteratedVariables):
    """Model predictive controller.

    .. versionadded:: >v4.5.1

        New interface to settings. The class has a property called ``settings`` which accesses an instance of :py:class:`MPCSettings` (please see this documentation for a list of available settings).
        Settings are now chosen as:

        ::

            mpc.settings.n_horizon = 20
        
        Previously, settings were passed to :py:meth:`set_param`. This method is still available and wraps the new interface.
        The new method has important advantages:
        
        1. The ``mpc.settings`` attribute can be printed to see the current configuration.

        2. Context help is available in most IDEs (e.g. VS Code) to see the available _settings, the type and a description.

        3. The :py:class:`MPCSettings` class has convenient methods, such as :py:meth:`MPCSettings.supress_ipopt_output()` to silence the solver.


    For general information on model predictive control, please read our `background article <../theory_mpc.html>`_ .

    The MPC controller extends the :py:class:`do_mpc.optimizer.Optimizer` base class
    (which is also used for the :py:class:`do_mpc.estimator.MHE` estimator).

    Use this class to configure and run the MPC controller
    based on a previously configured :py:class:`do_mpc.model.Model` instance.

    **Configuration and setup:**

    Configuring and setting up the MPC controller involves the following steps:

    1. Configure the MPC controller with :py:class:`MPCSettings`. The MPC instance has the attribute ``settings`` which is an instance of :py:class:`MPCSettings`. 

    2. Set the objective of the control problem with :py:func:`set_objective` and :py:func:`set_rterm`

    3. Set upper and lower bounds with :py:attr:`bounds` (optional).

    4. Set further (non-linear) constraints with :py:func:`set_nl_cons` (optional).

    5. Use the low-level API (:py:func:`get_p_template` and :py:func:`set_p_fun`) or high level API (:py:func:`set_uncertainty_values`) to create scenarios for robust MPC (optional).

    6. Use :py:meth:`get_tvp_template` and :py:meth:`set_tvp_fun` to create a method to obtain new time-varying parameters at each iteration.

    7. To finalize the class configuration there are two ways. The default approach is to call :py:meth:`setup`. For deep customization use the combination of :py:meth:`prepare_nlp` and :py:meth:`create_nlp`. See graph below for an illustration of the process.

    .. graphviz::
        :name: route_to_setup
        :caption: Route to setting up the MPC class.
        :align: center

        digraph G {
            graph [fontname = "helvetica"];
            rankdir=LR;

            subgraph cluster_main {
                node [fontname = "helvetica", shape=box, fontcolor="#404040", color="#707070"];
                edge [fontname = "helvetica", color="#707070"];

                start [label="Two ways to setup"];
                setup [label="setup", href="../api/do_mpc.controller.MPC.html#setup", target="_top", fontname = "Consolas"];
                create_nlp [label="create_nlp", href="../api/do_mpc.controller.MPC.html#create-nlp", target="_top", fontname = "Consolas"];
                process [label="Modify NLP"];
                prepare_nlp [label="prepare_nlp", href="../api/do_mpc.controller.MPC.html#prepare-nlp", target="_top", fontname = "Consolas"];
                finish [label="Configured MPC class"]
                start -> setup, prepare_nlp;
                prepare_nlp -> process;
                process -> create_nlp;
                setup, create_nlp -> finish;
                color=none;
            }

            subgraph cluster_modification {
                rankdir=TB;
                node [fontname = "helvetica", shape=box, fontcolor="#404040", color="#707070"];
                edge [fontname = "helvetica", color="#707070"];
                opt_x [label="opt_x", href="../api/do_mpc.controller.MPC.html#opt-x", target="_top", fontname = "Consolas"];
                opt_p [label="opt_p", href="../api/do_mpc.controller.MPC.html#opt-p", target="_top", fontname = "Consolas"];
                nlp_cons [label="nlp_cons", href="../api/do_mpc.controller.MPC.html#nlp-cons", target="_top", fontname = "Consolas"];
                nlp_obj [label="nlp_obj", href="../api/do_mpc.controller.MPC.html#nlp-obj", target="_top", fontname = "Consolas"];
                opt_x -> nlp_cons, nlp_obj;
                opt_p -> nlp_cons, nlp_obj;

                label = "Attributes to modify the NLP.";
		        color=black;
            }

            nlp_cons -> process;
            nlp_obj -> process;
        }

    Args:
        model: Model
        _settings: Settings for the MPC controller. See :py:class:`MPCSettings` for details.

    Warnings:
        Before running the controller, make sure to supply a valid initial guess for all optimized variables (states, algebraic states and inputs).
        Simply set the initial values of :py:attr:`x0`, :py:attr:`z0` and :py:attr:`u0` and then call :py:func:`set_initial_guess`.

        To take full control over the initial guess, modify the values of :py:attr:`opt_x_num`.

    During runtime call :py:func:`make_step` with the current state :math:`x` to get the optimal control input :math:`u`.

    """
    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel], settings: Optional[MPCSettings] = None):

        self.model = model

        assert model.flags['setup'] == True, 'Model for MPC was not setup. After the complete model creation call model.setup().'
        self.data = do_mpc.data.MPCData(self.model)
        self.data.dtype = 'MPC'

        # Initialize parent class:
        do_mpc.model.IteratedVariables.__init__(self)
        do_mpc.optimizer.Optimizer.__init__(self)

        # Initialize further structures specific to the MPC optimization problem.
        # This returns an identical numerical structure with all values set to the passed value.
        self._x_terminal_lb = model._x(-np.inf)
        self._x_terminal_ub = model._x(np.inf)

        self.rterm_factor = self.model._u(0.0)
        self.u_prev = self.copy_struct(self.model.u)

        # Initialize structure to hold the optimial solution and initial guess:
        self._opt_x_num = None
        # Initialize structure to hold the parameters for the optimization problem:
        self._opt_p_num = None

        # initialize MPC _settings class
        self._settings = MPCSettings()

        # Flags are checked when calling .setup.
        self.flags.update({
            'set_objective': False,
            'rterm_fun': False,
            'set_rterm': False,
            'set_tvp_fun': False,
            'set_p_fun': False,
            'set_initial_guess': False,
        })

    @property
    def settings(self):
        '''
        All necessary parameters of the mpc formulation.

        This is a core attribute of the MPC class. It is used to set and change parameters when setting up the controller
        by accessing an instance of :py:class:`MPCSettings`. 
        
        Example to change settings:

        ::

            MPC.settings.n_horizon = 15

        Note:     
            Settings cannot be updated after calling :py:meth:`do_mpc.controller.MPC.setup`.

        For a detailed list of all available parameters see :py:class:`MPCSettings`.
        '''
        return self._settings
    
    @settings.setter
    def settings(self, val):
        warnings.warn('Cannot change the settings attribute')

    @property
    def opt_x_num(self):
        """Full MPC solution and initial guess.

        This is the core attribute of the MPC class.
        It is used as the initial guess when solving the optimization problem
        and then overwritten with the current solution.

        The attribute is a CasADi numeric structure with nested power indices.
        It can be indexed as follows:

        ::

            # dynamic states:
            opt_x_num['_x', time_step, scenario, collocation_point, _x_name]
            # algebraic states:
            opt_x_num['_z', time_step, scenario, collocation_point, _z_name]
            # inputs:
            opt_x_num['_u', time_step, scenario, _u_name]
            # slack variables for soft constraints:
            opt_x_num['_eps', time_step, scenario, _nl_cons_name]

        The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
        Further indices are possible, if the variables are itself vectors or matrices.

        The attribute can be used **to manually set a custom initial guess or for debugging purposes**.

        **How to query?**

        Querying the structure is more complicated than it seems at first look because of the scenario-tree used
        for robust MPC. To obtain all collocation points for the finite element at time-step :math:`k` and scenario :math:`b` use:

        ::

            horzcat(*[mpc.opt_x_num['_x',k,b,-1]]+mpc.opt_x_num['_x',k+1,b,:-1])

        Due to the multi-stage formulation at any given time :math:`k` we can have multiple future scenarios.
        However, there is only exactly one scenario that lead to the current node in the tree.
        Thus the collocation points associated to the finite element :math:`k` lie in the past.

        The concept is illustrated in the figure below:

        .. figure:: ../static/collocation_points_scenarios.svg

        Note:
            The attribute ``opt_x_num`` carries the scaled values of all variables. See ``opt_x_num_unscaled``
            for the unscaled values (these are not used as the initial guess).

        Warnings:
            Do not tweak or overwrite this attribute unless you known what you are doing.

        Note:
            The attribute is populated when calling :py:func:`setup`
        """
        return self._opt_x_num

    @opt_x_num.setter
    def opt_x_num(self, val):
        self._opt_x_num = val

    @property
    def opt_p_num(self):
        """Full MPC parameter vector.

        This attribute is used when calling the MPC solver to pass all required parameters,
        including

        * initial state

        * uncertain scenario parameters

        * time-varying parameters

        * previous input sequence

        **do-mpc** handles setting these parameters automatically in the :py:func:`make_step`
        method. However, you can set these values manually and directly call :py:func:`solve`.

        The attribute is a CasADi numeric structure with nested power indices.
        It can be indexed as follows:

        ::

            # initial state:
            opt_p_num['_x0', _x_name]
            # uncertain scenario parameters
            opt_p_num['_p', scenario, _p_name]
            # time-varying parameters:
            opt_p_num['_tvp', time_step, _tvp_name]
            # input at time k-1:
            opt_p_num['_u_prev', time_step, scenario]

        The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
        Further indices are possible, if the variables are itself vectors or matrices.

        Warnings:
            Do not tweak or overwrite this attribute unless you known what you are doing.

        Note:
            The attribute is populated when calling :py:func:`setup`
        """
        return self._opt_p_num

    @opt_p_num.setter
    def opt_p_num(self, val):
        self._opt_p_num = val

    @property
    def opt_x(self):
        """Full structure of (symbolic) MPC optimization variables.

        The attribute is a CasADi symbolic structure with nested power indices.
        It can be indexed as follows:

        ::

            # dynamic states:
            opt_x['_x', time_step, scenario, collocation_point, _x_name]
            # algebraic states:
            opt_x['_z', time_step, scenario, collocation_point, _z_name]
            # inputs:
            opt_x['_u', time_step, scenario, _u_name]
            # slack variables for soft constraints:
            opt_x['_eps', time_step, scenario, _nl_cons_name]

        The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
        Further indices are possible, if the variables are itself vectors or matrices.

        The attribute can be used to alter the objective function or constraints of the NLP.

        **How to query?**

        Querying the structure is more complicated than it seems at first look because of the scenario-tree used
        for robust MPC. To obtain all collocation points for the finite element at time-step :math:`k` and scenario :math:`b` use:

        ::

            horzcat(*[mpc.opt_x['_x',k,b,-1]]+mpc.opt_x['_x',k+1,b,:-1])

        Due to the multi-stage formulation at any given time :math:`k` we can have multiple future scenarios.
        However, there is only exactly one scenario that lead to the current node in the tree.
        Thus the collocation points associated to the finite element :math:`k` lie in the past.

        The concept is illustrated in the figure below:

        .. figure:: ../static/collocation_points_scenarios.svg

        Note:
            The attribute ``opt_x`` carries the scaled values of all variables.

        Note:
            The attribute is populated when calling :py:func:`setup` or :py:func:`prepare_nlp`
        """
        return self._opt_x

    @opt_x.setter
    def opt_x(self, val):
        self._opt_x = val

    @property
    def opt_p(self):
        """Full structure of (symbolic) MPC parameters.

        The attribute is a CasADi numeric structure with nested power indices.
        It can be indexed as follows:

        ::

            # initial state:
            opt_p['_x0', _x_name]
            # uncertain scenario parameters
            opt_p['_p', scenario, _p_name]
            # time-varying parameters:
            opt_p['_tvp', time_step, _tvp_name]
            # input at time k-1:
            opt_p['_u_prev', time_step, scenario]

        The names refer to those given in the :py:class:`do_mpc.model.Model` configuration.
        Further indices are possible, if the variables are itself vectors or matrices.

        Warnings:
            Do not tweak or overwrite this attribute unless you known what you are doing.

        Note:
            The attribute is populated when calling :py:func:`setup` or :py:func:`prepare_nlp`
        """
        return self._opt_p

    @opt_p.setter
    def opt_p(self, val):
        self._opt_p = val

    @IndexedProperty
    def terminal_bounds(self, ind):
        """Query and set the terminal bounds for the states.
        The :py:func:`terminal_bounds` method is an indexed property, meaning
        getting and setting this property requires an index and calls this function.
        The power index (elements are seperated by commas) must contain at least the following elements:

        ======      =================   ==========================================================
        order       index name          valid options
        ======      =================   ==========================================================
        1           bound type          ``lower`` and ``upper``
        2           variable name       Names defined in :py:class:`do_mpc.model.Model`.
        ======      =================   ==========================================================

        Further indices are possible (but not neccessary) when the referenced variable is a vector or matrix.

        **Example**:

        ::

            # Set with:
            optimizer.terminal_bounds['lower', 'phi_1'] = -2*np.pi
            optimizer.terminal_bounds['upper', 'phi_1'] = 2*np.pi

            # Query with:
            optimizer.terminal_bounds['lower', 'phi_1']
        """
        assert isinstance(ind, tuple), 'Power index must include bound_type, var_name (as a tuple).'
        assert len(ind)>=2, 'Power index must include bound_type, var_type, var_name (as a tuple).'
        bound_type = ind[0]
        var_name   = ind[1:]

        err_msg = 'Invalid power index {} for bound_type. Must be from (lower, upper).'
        assert bound_type in ('lower', 'upper'), err_msg.format(bound_type)

        if bound_type == 'lower':
            query = '{var_type}_{bound_type}'.format(var_type="_x", bound_type='lb')
        elif bound_type == 'upper':
            query = '{var_type}_{bound_type}'.format(var_type="_x", bound_type='ub')
        # query results string e.g. _x_lb, _x_ub

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .bounds with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

        return var_struct[var_name]

    @terminal_bounds.setter
    def terminal_bounds(self, ind, val):
        """See Docstring for bounds getter method"""

        assert isinstance(ind, tuple), 'Power index must include bound_type, var_type, var_name (as a tuple).'
        assert len(ind)>=2, 'Power index must include bound_type, var_type, var_name (as a tuple).'
        bound_type = ind[0]
        var_name   = ind[1:]

        err_msg = 'Invalid power index {} for bound_type. Must be from (lower, upper).'
        assert bound_type in ('lower', 'upper'), err_msg.format(bound_type)

        if bound_type == 'lower':
            query = '_x_terminal_lb'
        elif bound_type == 'upper':
            query = '_x_terminal_ub'

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .bounds with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

        # Set value on struct:
        var_struct[var_name] = val

    def set_param(self, **kwargs)->None:
        """Set the parameters of the :py:class:`MPC` class. Parameters must be passed as pairs of valid keywords and respective argument.
        
        .. deprecated:: >v4.5.1
            This function will be deprecated in the future

        Note:
            A comprehensive list of all available parameters can be found in :py:class:`do_mpc.controller.MPCSettings` 
        
        For example:
        
        ::

            mpc._settings.n_horizon = 20
        
        The old interface, as shown in the example below, can still be accessed until further notice.
        
        ::

            mpc.set_param(n_horizon = 20)


        Note: 
            The only required parameters  are ``n_horizon`` and ``t_step``. All other parameters are optional.


        Note: 
            We highly suggest to change the linear solver for IPOPT from `mumps` to `MA27`. 
            Any available linear solver can be set using :py:meth:`do_mpc.controller.MPCSettings.set_linear_solver`.
            For more details, please check the :py:class:`do_mpc.controller.MPCSettings`.
        
        Note: 
            The output of IPOPT can be suppressed :py:meth:`do_mpc.controller.MPCSettings.supress_ipopt_output`.
            For more details, please check the :py:class:`do_mpc.controller.MPCSettings`.
        """
        assert self.flags['setup'] == False, 'Setting parameters after setup is prohibited.'

        for key, value in kwargs.items():
            if hasattr(self._settings, key):
                    setattr(self._settings, key, value)
            else:
                print('Warning: Key {} does not exist for MPC.'.format(key))

    def set_objective(self, mterm:Union[castools.SX,castools.MX]=None, lterm:Union[castools.SX,castools.MX]=None)->None:
        """Sets the objective of the optimal control problem (OCP). We introduce the following cost function:

        .. math::
           J(x,u,z) =  \\sum_{k=0}^{N}\\left(\\underbrace{l(x_k,z_k,u_k,p_k,p_{\\text{tv},k})}_{\\text{lagrange term}}
           + \\underbrace{\\Delta u_k^T R \\Delta u_k}_{\\text{r-term}}\\right)
           + \\underbrace{m(x_{N+1})}_{\\text{meyer term}}

        which is applied to the discrete-time model **AND** the discretized continuous-time model.
        For discretization we use `orthogonal collocation on finite elements`_ .
        The cost function is evaluated only on the first collocation point of each interval.

        .. _`orthogonal collocation on finite elements`: ../theory_orthogonal_collocation.html

        :py:func:`set_objective` is used to set the :math:`l(x_k,z_k,u_k,p_k,p_{\\text{tv},k})` (``lterm``) and :math:`m(x_{N+1})` (``mterm``), where ``N`` is the prediction horizon.
        Please see :py:func:`set_rterm` for the penalization of the control inputs.

        Args:
            lterm: Stage cost - **scalar** symbolic expression with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``
            mterm: Terminal cost - **scalar** symbolic expression with respect to ``_x`` and ``_p``

        Raises:
            assertion: mterm must have ``shape=(1,1)`` (scalar expression)
            assertion: lterm must have ``shape=(1,1)`` (scalar expression)
        """
        assert mterm.shape == (1,1), 'mterm must have shape=(1,1). You have {}'.format(mterm.shape)
        assert lterm.shape == (1,1), 'lterm must have shape=(1,1). You have {}'.format(lterm.shape)
        assert self.flags['setup'] == False, 'Cannot call .set_objective after .setup().'

        _x, _u, _z, _tvp, _p = self.model['x','u','z','tvp','p']


        # Check if mterm is valid:
        if not isinstance(mterm, (castools.DM, castools.SX, castools.MX)):
            raise Exception('mterm must be of type casadi.DM, casadi.SX or casadi.MX. You have: {}.'.format(type(mterm)))

        # Check if lterm is valid:
        if not isinstance(lterm, (castools.DM, castools.SX, castools.MX)):
            raise Exception('lterm must be of type casadi.DM, casadi.SX or casadi.MX. You have: {}.'.format(type(lterm)))

        if mterm is None:
            self.mterm = castools.DM(0)
        else:
            self.mterm = mterm
        # TODO: This function should be evaluated with scaled variables.
        self.mterm_fun = castools.Function('mterm', [_x, _tvp, _p], [mterm])

        if lterm is None:
            self.lterm = castools.DM(0)
        else:
            self.lterm = lterm

        self.lterm_fun = castools.Function('lterm', [_x, _u, _z, _tvp, _p], [lterm])

        # Check if lterm and mterm use invalid variables as inputs.
        # For the check we evaluate the function with dummy inputs and expect a DM output.
        err_msg = '{} contains invalid symbolic variables as inputs. Must contain only: {}'
        try:
            self.mterm_fun(_x(0),_tvp(0),_p(0))
        except:
            raise Exception(err_msg.format('mterm','_x, _tvp, _p'))
        try:
            self.lterm(_x(0),_u(0), _z(0), _tvp(0), _p(0))
        except:
            err_msg.format('lterm', '_x, _u, _z, _tvp, _p')

        self.flags['set_objective'] = True

    def set_rterm(self, rterm:Union[castools.SX,castools.MX]=None, **kwargs)->None:
        """Set the penality factor for the inputs. Call this function with keyword argument refering to the input names in
        :py:class:`model` and the penalty factor as the respective value.

        We define for :math:`i \\in \\mathbb{I}`, where :math:`\\mathbb{I}` is the set of inputs
        and all :math:`k=0,\\dots, N` where :math:`N` denotes the horizon:

        .. math::

            \\Delta u_{k,i} = u_{k,i} - u_{k-1,i}

        and add:

        .. math::

            \\sum_{k=0}^N \\sum_{i \\in \\mathbb{I}} r_{i}\\Delta u_{k,i}^2,

        the weighted squared cost to the MPC objective function.

        **Example:**

        ::

            # in model definition:
            Q_heat = model.set_variable(var_type='_u', var_name='Q_heat')
            F_flow = model.set_variable(var_type='_u', var_name='F_flow')

            ...
            # in MPC configuration:
            MPC.set_rterm(Q_heat = 10)
            MPC.set_rterm(F_flow = 10)
            # or alternatively:
            MPC.set_rterm(Q_heat = 10, F_flow = 10)

        In the above example we set :math:`r_{Q_{\\text{heat}}}=10`
        and :math:`r_{F_{\\text{flow}}}=10`.

        Note:
            As of version 4.6.3, set_rterm can be called with a user-defined penalty that overrides the default quadratic penalty term.
            Note that the inputs of the previous calculation step are stored in the mpc class and cannot be called from the model, see example.
            ``u_prev`` is generated automatically when the :py:class:`MPC` class is initialized.

        Args:
            rterm: Penalty term on input change - **scalar** symbolic expression with respect to ``_x``, ``_u``, ``_u_prev``, ``_u_prev``, ``_tvp``, ``_p``

        **Example:**

        ::

            # in model definition:
            Q_heat = model.set_variable(var_type='_u', var_name='Q_heat')
            F_flow = model.set_variable(var_type='_u', var_name='F_flow')

            ...
            # in MPC configuration:
            rterm = (model.u['Q_heat'] - mpc.u_prev['Q_heat'])**2 + (model.u['F_flow'] - mpc.u_prev['F_flow'])**2
            MPC.set_rterm(rterm)

        Note:
            For :math:`k=0` we obtain :math:`u_{-1}` from the previous solution.
        """
        assert self.flags['setup'] == False, 'Cannot call .set_rterm after .setup().'

        # User defined penalty term
        if rterm is not None:
            # Check if rterm is valid:
            assert rterm.shape == (1,1), 'rterm must have shape=(1,1). You have {}'.format(rterm.shape)
            
            if not isinstance(rterm, (castools.DM, castools.SX, castools.MX)):
                raise Exception('mterm must be of type casadi.DM, casadi.SX or casadi.MX. You have: {}.'.format(type(rterm)))
            
            _u_prev = self.u_prev
            _x, _u, _z, _tvp, _p = self.model['x','u','z','tvp','p']
            self.rterm_fun = castools.Function('rterm', [_x, _u, _u_prev, _z, _tvp, _p], [rterm])

            self.flags['rterm_fun'] = True
        
        # Default quadratic penalty term
        else:     
            for key, val in kwargs.items():
                assert key in self.model._u.keys(), 'Must pass keywords that refer to input names defined in model. Valid is: {}. You have: {}'.format(self.model._u.keys(), key)
                assert isinstance(val, (int, float, np.ndarray)), 'Value for {} must be int, float or numpy.ndarray. You have: {}'.format(key, type(val))
                self.rterm_factor[key] = val

        self.flags['set_rterm'] = True

    def copy_struct(self, original_struct):
        """
        Create a copy of a given CasADi struct.
        This method is called during initialization to copy the struct containing the system inputs ``u``.
        The copied structure is an identical copy of the input structure and is used in :py:func:`set_rterm` as a symbolic variable for past inputs.

        Args:
            original_struct: A CasADi struct (either SXStruct or MXStruct).

        Returns:
            A new CasADi struct with the same structure and entry names as the original.
        """
        struct_type = type(original_struct)

        if struct_type == castools.struct_symSX:
            sv = _SymVar('SX')
        elif struct_type == castools.struct_symMX:
            sv = _SymVar('MX')
        else:
            raise ValueError("Input is not a valid CasADi struct")

        var_dict = {
            'name': [name for name in original_struct.keys()],
            'var': [original_struct[name] for name in original_struct.keys()]
        }

        new_struct = sv.sym_struct([
            castools.entry(name, shape=var.shape) for var, name in zip(var_dict['var'], var_dict['name'])
        ])

        return new_struct

    def get_p_template(self, n_combinations:int)->None:
        """Obtain output template for :py:func:`set_p_fun`.

        Low level API method to set user defined scenarios for robust multi-stage MPC by defining an arbitrary number
        of combinations for the parameters defined in the model.
        For more details on robust multi-stage MPC please read our `background article <../theory_mpc.html#robust-multi-stage-nmpc>`_

        The method returns a structured object which is
        initialized with all zeros.
        Use this object to define values of the parameters for an arbitrary number of scenarios (defined by ``n_combinations``).

        This structure (with numerical values) should be used as the output of the ``p_fun`` function
        which is set to the class with :py:func:`set_p_fun`.

        Use the combination of :py:func:`get_p_template` and :py:func:`set_p_template` as a more adaptable alternative to :py:func:`set_uncertainty_values`.

        Note:
            We advice less experienced users to use :py:func:`set_uncertainty_values` as an alterntive way to configure the
            scenario-tree for robust multi-stage MPC.

        **Example:**

        ::

            # in model definition:
            alpha = model.set_variable(var_type='_p', var_name='alpha')
            beta = model.set_variable(var_type='_p', var_name='beta')

            ...
            # in MPC configuration:
            n_combinations = 3
            p_template = MPC.get_p_template(n_combinations)
            p_template['_p',0] = np.array([1,1])
            p_template['_p',1] = np.array([0.9, 1.1])
            p_template['_p',2] = np.array([1.1, 0.9])

            def p_fun(t_now):
                return p_template

            MPC.set_p_fun(p_fun)

        Note the nominal case is now:
        alpha = 1,
        beta = 1
        which is determined by the order in the arrays above (first element is nominal).

        Args:
            n_combinations: Define the number of combinations for the uncertain parameters for robust MPC.
        """
        self.n_combinations = n_combinations
        p_template = self.model.sv.sym_struct([
            castools.entry('_p', repeat=n_combinations, struct=self.model._p)
        ])
        return p_template(0)

    def set_p_fun(self, p_fun:Callable[[float],Union[castools.structure3.SXStruct,castools.structure3.MXStruct]])->None:
        """Set function which returns parameters.
        The ``p_fun`` is called at each optimization step to get the current values of the (uncertain) parameters.

        This is the low-level API method to set user defined scenarios for robust multi-stage MPC by defining an arbitrary number
        of combinations for the parameters defined in the model.
        For more details on robust multi-stage MPC please read our `background article <../theory_mpc.html#robust-multi-stage-nmpc>`_ .

        The method takes as input a function, which MUST
        return a structured object, based on the defined parameters and the number of combinations.
        The defined function has time as a single input.

        Obtain this structured object first, by calling :py:func:`get_p_template`.

        Use the combination of :py:func:`get_p_template` and :py:func:`set_p_fun` as a more adaptable alternative to :py:func:`set_uncertainty_values`.

        Note:
            We advice less experienced users to use :py:func:`set_uncertainty_values` as an alterntive way to configure the
            scenario-tree for robust multi-stage MPC.

        **Example:**

        ::

            # in model definition:
            alpha = model.set_variable(var_type='_p', var_name='alpha')
            beta = model.set_variable(var_type='_p', var_name='beta')

            ...
            # in MPC configuration:
            n_combinations = 3
            p_template = MPC.get_p_template(n_combinations)
            p_template['_p',0] = np.array([1,1])
            p_template['_p',1] = np.array([0.9, 1.1])
            p_template['_p',2] = np.array([1.1, 0.9])

            def p_fun(t_now):
                return p_template

            MPC.set_p_fun(p_fun)

        Note the nominal case is now:
        ``alpha = 1``,
        ``beta = 1``
        which is determined by the order in the arrays above (first element is nominal).

        Args:
            p_fun: Function which returns a structure with numerical values. Must be the same structure as obtained from :py:func:`get_p_template`. Function must have a single input (time).
        """
        assert self.get_p_template(self.n_combinations).labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.flags['set_p_fun'] = True
        self.p_fun = p_fun

    def set_uncertainty_values(self, **kwargs)->None:
        """Define scenarios for the uncertain parameters.
        High-level API method to conveniently set all possible scenarios for multistage MPC.
        For more details on robust multi-stage MPC please read our `background article <../theory_mpc.html#robust-multi-stage-nmpc>`_ .

        Pass a number of keyword arguments, where each keyword refers to a user defined parameter name from the model definition.
        The value for each parameter must be an array (or list), with an arbitrary number of possible values for this parameter.
        The first element is the nominal case.

        **Example:**

        ::

                # in model definition:
                alpha = model.set_variable(var_type='_p', var_name='alpha')
                beta = model.set_variable(var_type='_p', var_name='beta')
                gamma = model.set_variable(var_type='_p', var_name='gamma')
                ...
                # in MPC configuration:
                alpha_var = np.array([1., 0.9, 1.1])
                beta_var = np.array([1., 1.05])
                MPC.set_uncertainty_values(
                    alpha = alpha_var,
                    beta = beta_var
                )

        Note:
            Parameters that are not imporant for the MPC controller (e.g. MHE tuning matrices)
            can be ignored with the new interface (see ``gamma`` in the example above).


        Note the nominal case is now:
        ``alpha = 1``,
        ``beta = 1``
        which is determined by the order in the arrays above (first element is nominal).

        Args:
            kwargs: Arbitrary number of keyword arguments.
        """
        # If uncertainty values are passed as dictionary, extract values and keys:
        if not kwargs:
            return None
        assert isinstance(kwargs, dict), 'Pass keyword arguments, where each keyword refers to a user-defined parameter name.'
        names = [i for i in kwargs.keys()]
        valid_names = self.model.p.keys()
        err_msg = 'You passed keywords {}. Valid keywords are: {} (refering to user-defined parameter names).'
        assert set(names).issubset(set(valid_names)), err_msg.format(names, valid_names)
        values = kwargs.values()
        p_scenario = list(itertools.product(*values))
        n_combinations = len(p_scenario)
        p_template = self.get_p_template(n_combinations)

        if kwargs:
            # Dict case (only parameters with name are set):
            p_template['_p', :, names] = p_scenario
        else:
            # List case (assume ALL parameters are given ...)
            p_template['_p', :] = p_scenario

        def p_fun(t_now):
            return p_template

        self.set_p_fun(p_fun)

    def _check_validity(self):
        """Private method to be called in :py:func:`setup`. Checks if the configuration is valid and
        if the optimization problem can be constructed.
        Furthermore, default values are set if they were not configured by the user (if possible).
        Specifically, we set dummy values for the ``tvp_fun`` and ``p_fun`` if they are not present in the model.
        """
        # Objective mus be defined.
        if self.flags['set_objective'] == False:
            raise Exception('Objective is undefined. Please call .set_objective() prior to .setup().')
        # rterm should have been set (throw warning if not)
        if self.flags['set_rterm'] == False:
            warnings.warn('rterm was not set and defaults to zero. Changes in the control inputs are not penalized. Can lead to oscillatory behavior.')
            time.sleep(2)
        # tvp_fun must be set, if tvp are defined in model.
        if self.flags['set_tvp_fun'] == False and self.model._tvp.size > 0:
            raise Exception('You have not supplied a function to obtain the time-varying parameters defined in model. Use .set_tvp_fun() prior to setup.')
        # p_fun must be set, if p are defined in model.
        if self.flags['set_p_fun'] == False and self.model._p.size > 0:
            raise Exception('You have not supplied a function to obtain the parameters defined in model. Use .set_p_fun() (low-level API) or .set_uncertainty_values() (high-level API) prior to setup.')

        if np.any(self.rterm_factor.cat.full() < 0):
            warnings.warn('You have selected negative values for the rterm penalizing changes in the control input.')
            time.sleep(2)

        # Lower bounds should be lower than upper bounds:
        for lb, ub in zip([self._x_lb, self._u_lb, self._z_lb], [self._x_ub, self._u_ub, self._z_ub]):
            bound_check = lb.cat > ub.cat
            bound_fail = [label_i for i,label_i in enumerate(lb.labels()) if bound_check[i]]
            if np.any(bound_check):
                raise Exception('Your bounds are inconsistent. For {} you have lower bound > upper bound.'.format(bound_fail))

        # Are terminal bounds for the states set? If not use default values (unless MPC is setup to not use terminal bounds)
        if np.all(self._x_terminal_ub.cat == np.inf) and self._settings.use_terminal_bounds:
            self._x_terminal_ub = self._x_ub
        if np.all(self._x_terminal_lb.cat == -np.inf) and self._settings.use_terminal_bounds:
            self._x_terminal_lb = self._x_lb

        # Set dummy functions for tvp and p in case these parameters are unused.
        if 'tvp_fun' not in self.__dict__:
            _tvp = self.get_tvp_template()

            def tvp_fun(t): return _tvp
            self.set_tvp_fun(tvp_fun)

        if 'p_fun' not in self.__dict__:
            _p = self.get_p_template(1)

            def p_fun(t): return _p
            self.set_p_fun(p_fun)

    def setup(self)->None:
        """Setup the MPC class.
        Internally, this method will create the MPC optimization problem under consideration
        of the supplied dynamic model and the given :py:class:`MPC` class instance configuration.

        The :py:func:`setup` method can be called again after changing the configuration
        (e.g. adapting bounds) and will simply overwrite the previous optimization problem.

        Note:
            After this call, the :py:func:`solve` and :py:func:`make_step` method is applicable.

        Warnings:
            The :py:func:`setup` method may take a while depending on the size of your MPC problem.
            Note that especially for robust multi-stage MPC with a long robust horizon and many
            possible combinations of the uncertain parameters very large problems will arise.

            For more details on robust multi-stage MPC please read our `background article <../theory_mpc.html#robust-multi-stage-nmpc>`_ .

        """
        self.prepare_nlp()
        self.create_nlp()

    def set_initial_guess(self)->None:
        """Initial guess for optimization variables.
        Uses the current class attributes :py:attr:`x0`, :py:attr:`z0` and :py:attr:`u0` to create the initial guess.
        The initial guess is simply the initial values for all :math:`k=0,\dots,N` instances of :math:`x_k`, :math:`u_k` and :math:`z_k`.

        Warnings:
            If no initial values for :py:attr:`x0`, :py:attr:`z0` and :py:attr:`u0` were supplied during setup, these default to zero.

        Note:
            The initial guess is fully customizable by directly setting values on the class attribute:
            :py:attr:`opt_x_num`.
        """
        assert self.flags['setup'] == True, 'MPC was not setup yet. Please call MPC.setup().'

        self.opt_x_num['_x'] = self._x0.cat/self._x_scaling
        self.opt_x_num['_u'] = self._u0.cat/self._u_scaling
        self.opt_x_num['_z'] = self._z0.cat/self._z_scaling

        self.flags['set_initial_guess'] = True

    def make_step(self, x0:Union[np.ndarray,castools.DM])->np.ndarray:
        """Main method of the class during runtime. This method is called at each timestep
        and returns the control input for the current initial state :py:obj:`x0`.

        The method prepares the MHE by setting the current parameters, calls :py:func:`solve`
        and updates the :py:class:`do_mpc.data.Data` object.

        Args:
            x0: Current state of the system.

        Returns:
            u0
        """
        # Check setup.
        assert self.flags['setup'] == True, 'MPC was not setup yet. Please call MPC.setup().'

        # Check input type.
        if isinstance(x0, (np.ndarray, castools.DM)):
            pass
        elif isinstance(x0, castools.structure3.DMStruct):
            x0 = x0.cat
        else:
            raise Exception('Invalid type {} for x0. Must be {}'.format(type(x0), (np.ndarray, castools.DM, castools.structure3.DMStruct)))

        # Check input shape.
        n_val = np.prod(x0.shape)
        assert n_val == self.model.n_x, 'Wrong input with shape {}. Expected vector with {} elements'.format(n_val, self.model.n_x)
        # Check (once) if the initial guess was supplied.
        if not self.flags['set_initial_guess']:
            warnings.warn('Intial guess for the MPC was not set. The solver call is likely to fail.')
            time.sleep(5)
            # Since do-mpc is warmstarting, the initial guess will exist after the first call.
            self.flags['set_initial_guess'] = True

        # Get current tvp, p and time (as well as previous u)
        u_prev = self._u0
        tvp0 = self.tvp_fun(self._t0)
        p0 = self.p_fun(self._t0)
        t0 = self._t0

        # Set the current parameter struct for the optimization problem:
        self.opt_p_num['_x0'] = x0
        self.opt_p_num['_u_prev'] = u_prev
        self.opt_p_num['_tvp'] = tvp0['_tvp']
        self.opt_p_num['_p'] = p0['_p']
        # Solve the optimization problem (method inherited from optimizer)
        self.solve()

        # Extract solution:
        u0 = self.opt_x_num['_u', 0, 0]*self._u_scaling
        z0 = self.opt_x_num['_z', 0, 0, 0]*self._z_scaling
        aux0 = self.opt_aux_num['_aux', 0, 0]

        # Store solution:
        self.data.update(_x = x0)
        self.data.update(_u = u0)
        self.data.update(_z = z0)
        self.data.update(_tvp = tvp0['_tvp', 0])
        self.data.update(_p = p0['_p', 0])
        self.data.update(_time = t0)
        self.data.update(_aux = aux0)

        # Store additional information
        self.data.update(opt_p_num = self.opt_p_num)
        if self._settings.store_full_solution == True:
            opt_x_num_unscaled = self.opt_x_num_unscaled
            opt_aux_num = self.opt_aux_num
            self.data.update(_opt_x_num = opt_x_num_unscaled)
            self.data.update(_opt_aux_num = opt_aux_num)
        if self._settings.store_lagr_multiplier == True:
            lam_g_num = self.lam_g_num
            self.data.update(_lam_g_num = lam_g_num)
        if len(self._settings.store_solver_stats) > 0:
            solver_stats = self.solver_stats
            store_solver_stats = self._settings.store_solver_stats
            self.data.update(**{stat_i: value for stat_i, value in solver_stats.items() if stat_i in store_solver_stats})

        # Update initial
        self._t0 = self._t0 + self._settings.t_step
        self._x0.master = castools.DM(x0)
        self._u0.master = castools.DM(u0)
        self._z0.master = castools.DM(z0)

        # Return control input:
        return u0.full()

    def _update_bounds(self):
        """Private method to update the bounds of the optimization variables based on the current values defined with :py:attr:`scaling`.
        """
        if self._settings.cons_check_colloc_points:   # Constraints for all collocation points.
            # Dont bound the initial state
            self.lb_opt_x['_x', 1:self._settings.n_horizon] = self._x_lb.cat
            self.ub_opt_x['_x', 1:self._settings.n_horizon] = self._x_ub.cat

            # Bounds for the algebraic variables:
            self.lb_opt_x['_z'] = self._z_lb.cat
            self.ub_opt_x['_z'] = self._z_ub.cat

            # Terminal bounds
            self.lb_opt_x['_x', self._settings.n_horizon, :, -1] = self._x_terminal_lb.cat
            self.ub_opt_x['_x', self._settings.n_horizon, :, -1] = self._x_terminal_ub.cat
        else:   # Constraints only at the beginning of the finite Element
            # Dont bound the initial state
            self.lb_opt_x['_x', 1:self._settings.n_horizon, :, -1] = self._x_lb.cat
            self.ub_opt_x['_x', 1:self._settings.n_horizon, :, -1] = self._x_ub.cat

            # Bounds for the algebraic variables:
            self.lb_opt_x['_z', :, :, 0] = self._z_lb.cat
            self.ub_opt_x['_z', :, : ,0] = self._z_ub.cat

            # Terminal bounds
            self.lb_opt_x['_x', self._settings.n_horizon, :, -1] = self._x_terminal_lb.cat
            self.ub_opt_x['_x', self._settings.n_horizon, :, -1] = self._x_terminal_ub.cat

        # Bounds for the inputs along the horizon
        self.lb_opt_x['_u'] = self._u_lb.cat
        self.ub_opt_x['_u'] = self._u_ub.cat

        # Bounds for the slack variables:
        self.lb_opt_x['_eps'] = self._eps_lb.cat
        self.ub_opt_x['_eps'] = self._eps_ub.cat

    def _prepare_nlp(self)->None:
        """Internal method. See detailed documentation with optimizer.prepare_nlp
        """
        self._settings.check_for_mandatory_settings()
        nl_cons_input = self.model['x', 'u', 'z', 'tvp', 'p']
        self._setup_nl_cons(nl_cons_input)
        self._check_validity()

        # Obtain an integrator (collocation, discrete-time) and the amount of intermediate (collocation) points
        ifcn, n_total_coll_points = self._setup_discretization()
        n_branches, n_scenarios, child_scenario, parent_scenario, branch_offset = self._setup_scenario_tree()

        # How many scenarios arise from the scenario tree (robust multi-stage MPC)
        n_max_scenarios = self.n_combinations ** self._settings.n_robust

        # If open_loop option is active, all scenarios (at a given stage) have the same input.
        if self._settings.open_loop:
            n_u_scenarios = 1
        else:
            # Else: Each scenario has its own input.
            n_u_scenarios = n_max_scenarios

        # How many slack variables (for soft constraints) are introduced over the horizon.
        if self._settings.nl_cons_single_slack:
            n_eps = 1
        else:
            n_eps = self._settings.n_horizon

        # Create struct for optimization variables:
        self._opt_x = opt_x = self.model.sv.sym_struct([
            # One additional point (in the collocation dimension) for the final point.
            castools.entry('_x', repeat=[self._settings.n_horizon+1, n_max_scenarios,
                                1+n_total_coll_points], struct=self.model._x),
            castools.entry('_z', repeat=[self._settings.n_horizon, n_max_scenarios,
                                max(n_total_coll_points,1)], struct=self.model._z),
            castools.entry('_u', repeat=[self._settings.n_horizon, n_u_scenarios], struct=self.model._u),
            castools.entry('_eps', repeat=[n_eps, n_max_scenarios], struct=self._eps),
        ])
        self.n_opt_x = self._opt_x.shape[0]
        # NOTE: The entry _x[k,child_scenario[k,s,b],:] starts with the collocation points from s to b at time k
        #       and the last point contains the child node
        # NOTE: Currently there exist dummy collocation points for the initial state (for each branch)

        # Create scaling struct as assign values for _x, _u, _z.
        self.opt_x_scaling = opt_x_scaling = opt_x(1)
        opt_x_scaling['_x'] = self._x_scaling
        opt_x_scaling['_z'] = self._z_scaling
        opt_x_scaling['_u'] = self._u_scaling
        # opt_x are unphysical (scaled) variables. opt_x_unscaled are physical (unscaled) variables.
        self.opt_x_unscaled = opt_x_unscaled = opt_x(opt_x.cat * opt_x_scaling)

        # Create struct for optimization parameters:
        self._opt_p = opt_p = self.model.sv.sym_struct([
            castools.entry('_x0', struct=self.model._x),
            castools.entry('_tvp', repeat=self._settings.n_horizon+1, struct=self.model._tvp),
            castools.entry('_p', repeat=self.n_combinations, struct=self.model._p),
            castools.entry('_u_prev', struct=self.model._u),
        ])
        _w = self.model._w(0)

        self.n_opt_p = opt_p.shape[0]

        # Dummy struct with symbolic variables
        self.aux_struct = self.model.sv.sym_struct([
            castools.entry('_aux', repeat=[self._settings.n_horizon, n_max_scenarios], struct=self.model._aux_expression)
        ])
        # Create mutable symbolic expression from the struct defined above.
        self._opt_aux = opt_aux = self.model.sv.struct(self.aux_struct)

        self.n_opt_aux = opt_aux.shape[0]

        self._lb_opt_x = opt_x(-np.inf)
        self._ub_opt_x = opt_x(np.inf)

        # Initialize objective function and constraints
        obj = castools.DM(0)
        cons = []
        cons_lb = []
        cons_ub = []

        # Initial condition:
        cons.append(opt_x['_x', 0, 0, -1]-opt_p['_x0']/self._x_scaling)

        cons_lb.append(np.zeros((self.model.n_x, 1)))
        cons_ub.append(np.zeros((self.model.n_x, 1)))

        # NOTE: Weigthing factors for the tree assumed equal. They could be set from outside
        # Weighting factor for every scenario
        omega = [1. / n_scenarios[k + 1] for k in range(self._settings.n_horizon)]
        omega_delta_u = [1. / n_scenarios[k + 1] for k in range(self._settings.n_horizon)]

        # For all control intervals
        for k in range(self._settings.n_horizon):
            # For all scenarios (grows exponentially with n_robust)
            for s in range(n_scenarios[k]):
                # For all childen nodes of each node at stage k, discretize the model equations

                # Scenario index for u is always 0 if self.open_loop = True
                s_u = 0 if self._settings.open_loop else s
                for b in range(n_branches[k]):
                    # Obtain the index of the parameter values that should be used for this scenario
                    current_scenario = b + branch_offset[k][s]

                    # Compute constraints and predicted next state of the discretization scheme
                    col_xk = castools.vertcat(*opt_x['_x', k+1, child_scenario[k][s][b], :-1])
                    col_zk = castools.vertcat(*opt_x['_z', k, child_scenario[k][s][b]])
                    [g_ksb, xf_ksb] = ifcn(opt_x['_x', k, s, -1], col_xk,
                                           opt_x['_u', k, s_u], col_zk, opt_p['_tvp', k],
                                           opt_p['_p', current_scenario], _w)

                    # Add the collocation equations
                    cons.append(g_ksb)
                    cons_lb.append(np.zeros(g_ksb.shape[0]))
                    cons_ub.append(np.zeros(g_ksb.shape[0]))

                    # Add continuity constraints
                    cons.append(xf_ksb - opt_x['_x', k+1, child_scenario[k][s][b], -1])
                    cons_lb.append(np.zeros((self.model.n_x, 1)))
                    cons_ub.append(np.zeros((self.model.n_x, 1)))

                    k_eps = min(k, n_eps-1)
                    if self._settings.nl_cons_check_colloc_points:
                        # Ensure nonlinear constraints on all collocation points
                        for i in range(n_total_coll_points):
                            nl_cons_k = self._nl_cons_fun(
                                opt_x_unscaled['_x', k+1, s, i], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, i],
                                opt_p['_tvp', k], opt_p['_p', current_scenario], opt_x_unscaled['_eps', k_eps, s])
                            cons.append(nl_cons_k)
                            cons_lb.append(self._nl_cons_lb)
                            cons_ub.append(self._nl_cons_ub)
                    else:
                        # Ensure nonlinear constraints only on the beginning of the FE
                        nl_cons_k = self._nl_cons_fun(
                            opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, 0],
                            opt_p['_tvp', k], opt_p['_p', current_scenario], opt_x_unscaled['_eps', k_eps, s])
                        cons.append(nl_cons_k)
                        cons_lb.append(self._nl_cons_lb)
                        cons_ub.append(self._nl_cons_ub)

                    # Add terminal constraints
                    # TODO: Add terminal constraints with an additional nl_cons

                    # Add contribution to the cost
                    obj += omega[k] * self.lterm_fun(opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u],
                                                     opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])
                    # Add slack variables to the cost
                    obj += self.epsterm_fun(opt_x_unscaled['_eps', k_eps, s])

                    # In the last step add the terminal cost too
                    if k == self._settings.n_horizon - 1:
                        obj += omega[k] * self.mterm_fun(opt_x_unscaled['_x', k + 1, s, -1], opt_p['_tvp', k+1],
                                                         opt_p['_p', current_scenario])

                    # U regularization:
                    # For user defined penalty term
                    if self.flags['rterm_fun'] == True:
                        if k==0:
                            obj += self.rterm_fun(opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_p['_u_prev']/self._u_scaling,
                                                     opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])
                        else:
                            obj += self.rterm_fun(opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_x['_u', k-1, parent_scenario[k][s_u]],
                                                     opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])
                    # Default penalty term
                    if self.flags['rterm_fun'] == False:
                        if k == 0:
                            obj += self.rterm_factor.cat.T@((opt_x['_u', 0, s_u]-opt_p['_u_prev']/self._u_scaling)**2)
                        else:
                            obj += self.rterm_factor.cat.T@((opt_x['_u', k, s_u]-opt_x['_u', k-1, parent_scenario[k][s_u]])**2)

                    # Calculate the auxiliary expressions for the current scenario:
                    opt_aux['_aux', k, s] = self.model._aux_expression_fun(
                        opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])

                    # For some reason when working with MX, the "unused" aux values in the scenario tree must be set explicitly (they are not ever used...)
                for s_ in range(n_scenarios[k],n_max_scenarios):
                    opt_aux['_aux', k, s_] = self.model._aux_expression_fun(
                        opt_x_unscaled['_x', k, s, -1], opt_x_unscaled['_u', k, s_u], opt_x_unscaled['_z', k, s, -1], opt_p['_tvp', k], opt_p['_p', current_scenario])

        # Set bounds for all optimization variables
        self._update_bounds()

        # Write all created elements to self:
        self._nlp_obj = obj
        self._nlp_cons = cons
        self._nlp_cons_lb = cons_lb
        self._nlp_cons_ub = cons_ub

        # Initialize copies of structures with numerical values (all zero):
        self._opt_x_num = self._opt_x(0)
        self.opt_x_num_unscaled = self._opt_x(0)
        self._opt_p_num = self._opt_p(0)
        self.opt_aux_num = self._opt_aux(0)

        self.flags['prepare_nlp'] = True

    def _create_nlp(self)->None:
        """Internal method. See detailed documentation in optimizer.create_nlp
        """
        self._nlp_cons = castools.vertcat(*self._nlp_cons)
        self._nlp_cons_lb = castools.vertcat(*self._nlp_cons_lb)
        self._nlp_cons_ub = castools.vertcat(*self._nlp_cons_ub)

        self.n_opt_lagr = self._nlp_cons.shape[0]
        # Create casadi optimization object:
        nlpsol_opts = {
            'expand': False,
            'ipopt.linear_solver': 'mumps',
        }.update(self._settings.nlpsol_opts)
        self.nlp = {'x': castools.vertcat(self._opt_x), 'f': self._nlp_obj, 'g': self._nlp_cons, 'p': castools.vertcat(self._opt_p)}
        self.S = castools.nlpsol('S', 'ipopt', self.nlp, self._settings.nlpsol_opts)

        # Create function to caculate all auxiliary expressions:
        self.opt_aux_expression_fun = castools.Function('opt_aux_expression_fun', [self._opt_x, self._opt_p], [self._opt_aux])

        # Gather meta information:
        meta_data = {key: getattr(self._settings, key) for key in asdict(self._settings).keys()}
        meta_data.update({'structure_scenario': self.scenario_tree['structure_scenario']})
        self.data.set_meta(**meta_data)

        self._prepare_data()

        self.flags['setup'] = True
