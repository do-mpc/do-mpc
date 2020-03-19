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
import itertools
import time

from do_mpc.tools.indexedproperty import IndexedProperty


class Optimizer:
    """The base clase for the optimization based state estimation (MHE) and predictive controller (MPC).
    This class establishes the jointly used attributes, methods and properties.

    .. warning::

        The ``Optimizer`` base class can not be used independently.

    """
    def __init__(self):
        assert 'model' in self.__dict__.keys(), 'Cannot initialize the optimizer before assigning the model to the current class instance.'

        # Initialize structures for bounds, scaling, initial values by calling the symbolic structures defined in the model
        # with the default numerical value.
        # This returns an identical numerical structure with all values set to the passed value.

        self._x_lb = self.model._x(-np.inf)
        self._x_ub = self.model._x(np.inf)

        self._u_lb = self.model._u(-np.inf)
        self._u_ub = self.model._u(np.inf)

        self._z_lb = self.model._z(-np.inf)
        self._z_ub = self.model._z(np.inf)

        self._x_scaling = self.model._x(1.0)
        self._u_scaling = self.model._u(1.0)
        self._z_scaling = self.model._z(1.0)
        self._p_scaling = self.model._p(1.0) # only meaningful for MHE.

        # Lists for further non-linear constraints (optional). Constraints are formulated as cons < ub
        self.nl_cons_list = [
            {'expr_name': 'default', 'expr': DM(), 'ub': DM()}
        ]
        self.slack_vars_list = [
            {'slack_name': 'default', 'var':SX.sym('default',(0,0)), 'ub': DM()}
        ]
        self.slack_cost = 0


    @IndexedProperty
    def bounds(self, ind):
        """Queries and sets the bounds of the optimization variables for the optimizer.
        The :py:func:`Optimizer.bounds` method is an indexed property, meaning
        getting and setting this property requires an index and calls this function.
        The power index (elements are seperated by comas) must contain atleast the following elements:

        ======      =================   ==========================================================
        order       index name          valid options
        ======      =================   ==========================================================
        1           bound type          ``lower`` and ``upper``
        2           variable type       ``_x``, ``_u`` and ``_z`` (and ``_p_est`` for MHE)
        3           variable name       Names defined in :py:class:`do_mpc.model`.
        ======      =================   ==========================================================

        Further indices are possible (but not neccessary) when the referenced variable is a vector or matrix.

        **Example**:

        ::

            # Set with:
            optimizer.bounds['lower','_x', 'phi_1'] = -2*np.pi
            optimizer.bounds['upper','_x', 'phi_1'] = 2*np.pi

            # Query with:
            optimizer.bounds['lower','_x', 'phi_1']

        """
        assert isinstance(ind, tuple), 'Power index must include bound_type, var_type, var_name (as a tuple).'
        assert len(ind)>=3, 'Power index must include bound_type, var_type, var_name (as a tuple).'
        bound_type = ind[0]
        var_type   = ind[1]
        var_name   = ind[2:]

        err_msg = 'Invalid power index {} for bound_type. Must be from (lower, upper).'
        assert bound_type in ('lower', 'upper'), err_msg.format(bound_type)
        err_msg = 'Invalid power index {} for var_type. Must be from (_x, states, _u, inputs, _z, algebraic).'
        assert var_type in ('_x', '_u', '_z', '_p_est'), err_msg.format(var_type)

        if bound_type == 'lower':
            query = '{var_type}_{bound_type}'.format(var_type=var_type, bound_type='lb')
        elif bound_type == 'upper':
            query = '{var_type}_{bound_type}'.format(var_type=var_type, bound_type='ub')
        # query results string e.g. _x_lb, _x_ub, _u_lb, u_ub ....

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .bounds with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), msg.format(ind, var_struct.keys())

        return var_struct[var_name]

    @bounds.setter
    def bounds(self, ind, val):
        """See Docstring for bounds getter method"""

        assert isinstance(ind, tuple), 'Power index must include bound_type, var_type, var_name (as a tuple).'
        assert len(ind)>=3, 'Power index must include bound_type, var_type, var_name (as a tuple).'
        bound_type = ind[0]
        var_type   = ind[1]
        var_name   = ind[2:]

        err_msg = 'Invalid power index {} for bound_type. Must be from (lower, upper).'
        assert bound_type in ('lower', 'upper'), err_msg.format(bound_type)
        err_msg = 'Invalid power index {} for var_type. Must be from (_x, _u, _z, _p_est).'
        assert var_type in ('_x', '_u', '_z', '_p_est'), err_msg.format(var_type)

        if bound_type == 'lower':
            query = '{var_type}_{bound_type}'.format(var_type=var_type, bound_type='lb')
        elif bound_type == 'upper':
            query = '{var_type}_{bound_type}'.format(var_type=var_type, bound_type='ub')
        # query results string e.g. _x_lb, _x_ub, _u_lb, u_ub ....

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .bounds with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), msg.format(ind, var_struct.keys())

        # Set value on struct:
        var_struct[var_name] = val


    @IndexedProperty
    def scaling(self, ind):
        """Queries and sets the scaling of the optimization variables for the optimizer.
        The :py:func:`Optimizer.scaling` method is an indexed property, meaning
        getting and setting this property requires an index and calls this function.
        The power index (elements are seperated by comas) must contain atleast the following elements:

        ======      =================   ==========================================================
        order       index name          valid options
        ======      =================   ==========================================================
        1           variable type       ``_x``, ``_u`` and ``_z`` (and ``_p_est`` for MHE)
        2           variable name       Names defined in :py:class:`do_mpc.model`.
        ======      =================   ==========================================================

        Further indices are possible (but not neccessary) when the referenced variable is a vector or matrix.

        **Example**:

        ::

            # Set with:
            optimizer.scaling['_x', 'phi_1'] = 2
            optimizer.scaling['_x', 'phi_2'] = 2

            # Query with:
            optimizer.scaling['_x', 'phi_1']

        .. note::

            Scaling the optimization problem is suggested when states and / or inputs take on values
            which differ by orders of magnitude.

        """
        assert isinstance(ind, tuple), 'Power index must include bound_type, var_type, var_name (as a tuple).'
        assert len(ind)>=2, 'Power index must include bound_type, var_type, var_name (as a tuple).'
        var_type   = ind[0]
        var_name   = ind[1:]

        err_msg = 'Invalid power index {} for var_type. Must be from (_x, states, _u, inputs, _z, algebraic).'
        assert var_type in ('_x', '_u', '_z', '_p_est'), err_msg.format(var_type)

        query = '{var_type}_scaling'.format(var_type=var_type)
        # query results string e.g. _x_scaling, _u_scaling

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .scaling with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), msg.format(ind, var_struct.keys())

        return var_struct[var_name]


    @scaling.setter
    def scaling(self, ind, val):
        """See Docstring for scaling getter method"""
        assert isinstance(ind, tuple), 'Power index must include bound_type, var_type, var_name (as a tuple).'
        assert len(ind)>=2, 'Power index must include bound_type, var_type, var_name (as a tuple).'
        var_type   = ind[0]
        var_name   = ind[1:]

        err_msg = 'Invalid power index {} for var_type. Must be from (_x, states, _u, inputs, _z, algebraic).'
        assert var_type in ('_x', '_u', '_z', '_p_est'), err_msg.format(var_type)

        query = '{var_type}_scaling'.format(var_type=var_type)
        # query results string e.g. _x_scaling, _u_scaling

        # Get the desired struct:
        var_struct = getattr(self, query)

        err_msg = 'Calling .scaling with {} is not valid. Possible keys are {}.'
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), msg.format(ind, var_struct.keys())

        var_struct[var_name] = val

    def set_initial_state(self, x0, p_est0=None, reset_history=False, set_intial_guess=True):
        """Set the intial state of the optimizer.
        Optionally resets the history. The history is empty upon creation of the optimizer.

        Optionally update the initial guess. The initial guess is first created with the ``.setup()`` method (MHE/MPC)
        and uses the class attributes ``_x0``, ``_u0``, ``_z0`` for all time instances, collocation points (if applicable)
        and scenarios (if applicable). If these values were not explicitly set by the user, they default to all zeros.


        :param x0: Initial state
        :type x0: numpy array
        :param reset_history: Resets the history of the optimizer, defaults to False
        :type reset_history: bool (,optional)
        :param set_intial_guess: Setting the initial state also updates the intial guess for the optimizer.
        :type set_intial_guess: bool (,optional)

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

        if p_est0 is not None:
            if isinstance(p_est0, (np.ndarray, casadi.DM)):
                self._p_est0 = self._p_est(p_est0)
            elif isinstance(p_est0, structure3.DMStruct):
                self._p_est0 = p_est0
            else:
                raise Exception('p_est0 must be of type (np.ndarray, casadi.DM, structure3.DMStruct). You have: {}'.format(type(p_est0)))

        if reset_history:
            self.reset_history()

        if set_intial_guess:
            self.set_initial_guess()

    def reset_history(self):
        """Reset the history of the optimizer.
        All data from the :py:class:`do_mpc.data.Data` instance is removed.
        """
        self.data.init_storage()
        self._t0 = np.array([0])

    def _prepare_data(self):
        """Write optimizer meta data to data object (all params set in self.data_fields).
        If selected, initialize the container for the full solution of the optimizer.
        """
        self.data.data_fields.update({'_eps': self.n_eps})
        self.data.data_fields.update({'opt_p_num': self.n_opt_p})
        self.data.opt_p = self.opt_p

        if self.store_full_solution == True:
            # Create data_field for the optimal solution.
            self.data.data_fields.update({'_opt_x_num': self.n_opt_x})
            self.data.data_fields.update({'_opt_aux_num': self.n_opt_aux})
            self.data.opt_x = self.opt_x
            # aux_struct is the struct_symSX variant of opt_aux (which is struct_SX). struct_SX cannot be unpickled (bug).
            # See: https://groups.google.com/forum/#!topic/casadi-users/dqAb4tnA2ik
            self.data.opt_aux = self.aux_struct
        if self.store_lagr_multiplier == True:
            # Create data_field for the lagrange multipliers
            self.data.data_fields.update({'_lam_g_num': self.n_opt_lagr})
        if len(self.store_solver_stats) > 0:
            # These are valid arguments for solver stats:
            solver_stats = ['iter_count', 'iterations', 'n_call_S', 'n_call_callback_fun',
                            'n_call_nlp_f', 'n_call_nlp_g', 'n_call_nlp_grad', 'n_call_nlp_grad_f',
                            'n_call_nlp_hess_l', 'n_call_nlp_jac_g', 'return_status', 'success', 't_proc_S',
                            't_proc_callback_fun', 't_proc_nlp_f', 't_proc_nlp_g', 't_proc_nlp_grad',
                            't_proc_nlp_grad_f', 't_proc_nlp_hess_l', 't_proc_nlp_jac_g', 't_wall_S',
                            't_wall_callback_fun', 't_wall_nlp_f', 't_wall_nlp_g', 't_wall_nlp_grad', 't_wall_nlp_grad_f',
                            't_wall_nlp_hess_l', 't_wall_nlp_jac_g']
            # Create data_field(s) for the recorded (valid) stats.
            for stat_i in self.store_solver_stats:
                assert stat_i in solver_stats, 'The requested {} is not a valid solver stat and cannot be recorded. Please supply one of the following (or none): {}'.format(stat_i, solver_stats)
                self.data.data_fields.update({stat_i: 1})

        self.data.init_storage()

    def set_nl_cons(self, expr_name, expr, ub=np.inf, soft_constraint=False, penalty_term_cons=1, maximum_violation=np.inf):
        """Introduce new constraint to the class. Further constraints are optional.
        Expressions must be formulated with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        They are implemented as:

        .. math::

            m(x,u,z,p_{\\text{tv}}, p) \\leq m_{\\text{ub}}

        Setting the flag ``soft_constraint=True`` will introduce slack variables :math:`\\epsilon`, such that:

        .. math::

            m(x,u,z,p_{\\text{tv}}, p)-\\epsilon &\\leq m_{\\text{ub}},\\\\
            0 &\\leq \\epsilon \\leq \\epsilon_{\\text{max}},

        Slack variables are added to the cost function and multiplied with the supplied penalty term.
        This formulation makes constraints soft, meaning that a certain violation is tolerated and does not lead to infeasibility.
        Typically, high values for the penalty are suggested to avoid significant violation of the constraints.

        :param expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX

        :raises assertion: expr_name must be str
        :raises assertion: expr must be a casadi SX or MX type

        :return: Returns the newly created expression. Expression can be used e.g. for the RHS.
        :rtype: casadi.SX
        """
        assert self.flags['setup'] == False, 'Cannot call .set_expression after .setup_model.'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have: {}'.format(type(expr))
        assert isinstance(ub, (int, float, np.ndarray)), 'ub must be float, int or numpy.ndarray, you have: {}'.format(type(ub))
        assert isinstance(soft_constraint, bool), 'soft_constraint must be boolean, you have: {}'.format(type(soft_constraint))

        if soft_constraint==True:
            # Introduce new slack variable:
            epsilon = SX.sym('eps_'+expr_name,*expr.shape)
            # Change expression
            expr = expr-epsilon
            # Add slack variable to list of slack variables:
            self.slack_vars_list.extend([
                {'slack_name': expr_name, 'var': epsilon, 'ub': maximum_violation}
            ])
            # Add cost contribution:
            self.slack_cost += sum1(penalty_term_cons*epsilon)


        self.nl_cons_list.extend([
            {'expr_name': expr_name, 'expr': expr, 'ub' : ub}])

        return expr

    def _setup_nl_cons(self):
        """Private method that is called from :py:func:`do_mpc.controller.MPC.setup` or :py:func:`do_mpc.estimator.MHE.setup`.
        Afterwards no further non-linear constraints can be added with the :py:func:`Optimizer.set_nl_cons` method.

        This is not part of the public API. Do not call this method.
        """
        # Create struct for soft constraints:
        self._eps = _eps = struct_symSX([
            entry(slack_i['slack_name'], sym=slack_i['var']) for slack_i in self.slack_vars_list
        ])
        self.n_eps = _eps.shape[0]

        # Create bounds:
        self._eps_lb = _eps(0.0)
        self._eps_ub = _eps(np.inf)
        # Set bounds:
        for slack_i in self.slack_vars_list:
            self._eps_ub[slack_i['slack_name']] = slack_i['ub']
        # Objective function epsilon contribution:
        self.epsterm_fun = Function('epsterm', [_eps], [self.slack_cost])

        # Create struct for _nl_cons:
        # Use the previously defined SX.sym variables to declare shape and symbolic variable.
        self._nl_cons = struct_SX([
            entry(expr_i['expr_name'], expr=expr_i['expr']) for expr_i in self.nl_cons_list
        ])
        # Make function from these expressions:
        _x, _u, _z, _tvp, _p = self.model['x', 'u', 'z', 'tvp', 'p']
        self._nl_cons_fun = Function('nl_cons_fun', [_x, _u, _z, _tvp, _p, _eps], [self._nl_cons])
        # Create bounds:
        self._nl_cons_ub = self._nl_cons(np.inf)
        self._nl_cons_lb = self._nl_cons(-np.inf)
        # Set bounds:
        for nl_cons_i in self.nl_cons_list:
            self._nl_cons_ub[nl_cons_i['expr_name']] = nl_cons_i['ub']


    def get_tvp_template(self):
        """The method returns a structured object with n_horizon elements, and a set of time varying parameters (as defined in model)
        for each of these instances. The structure is initialized with all zeros. Use this object to define values of the time varying parameters.

        This structure (with numerical values) should be used as the output of the ``tvp_fun`` function which is set to the class with :py:func:`Optimizer.set_tvp_fun`.
        Use the combination of :py:func:`Optimizer.get_tvp_template` and :py:func:`Optimizer.set_tvp_fun`.

        Example:

        ::

            # in model definition:
            alpha = model.set_variable(var_type='_tvp', var_name='alpha')
            beta = model.set_variable(var_type='_tvp', var_name='beta')

            ...
            # in optimizer configuration:
            tvp_temp_1 = optimizer.get_tvp_template()
            tvp_temp_1['_tvp', :] = np.array([1,1])

            tvp_temp_2 = optimizer.get_tvp_template()
            tvp_temp_2['_tvp', :] = np.array([0,0])

            def tvp_fun(t_now):
                if t_now<10:
                    return tvp_temp_1
                else:
                    tvp_temp_2

            optimizer.set_tvp_fun(tvp_fun)

        :return: None
        :rtype: None
        """

        tvp_template = struct_symSX([
            entry('_tvp', repeat=self.n_horizon, struct=self.model._tvp)
        ])
        return tvp_template(0)

    def set_tvp_fun(self, tvp_fun):
        """ Set the tvp_fun which is called at each optimization step to get the current prediction of the time-varying parameters.
        The supplied function must be callable with the current time as the only input. Furthermore, the function must return
        a CasADi structured object which is based on the horizon and on the model definition. The structure can be obtained with
        :py:func:`Optimizer.get_tvp_template`.

        ::

            # in model definition:
            alpha = model.set_variable(var_type='_tvp', var_name='alpha')
            beta = model.set_variable(var_type='_tvp', var_name='beta')

            ...
            # in optimizer configuration:
            tvp_temp_1 = optimizer.get_tvp_template()
            tvp_temp_1['_tvp', :] = np.array([1,1])

            tvp_temp_2 = optimizer.get_tvp_template()
            tvp_temp_2['_tvp', :] = np.array([0,0])

            def tvp_fun(t_now):
                if t_now<10:
                    return tvp_temp_1
                else:
                    tvp_temp_2

            optimizer.set_tvp_fun(tvp_fun)

        The method :py:func:`Optimizer.set_tvp_fun`. must be called prior to setup IF time-varying parameters are defined in the model.

        :param tvp_fun: Function that returns the predicted tvp values at each timestep. Must have single input (float) and return a structure3.DMStruct (obtained with .get_tvp_template())
        :type tvp_fun: function

        """
        assert isinstance(tvp_fun(0), structure3.DMStruct), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'

        self.flags['set_tvp_fun'] = True

        self.tvp_fun = tvp_fun

    def solve(self):
        """Solves the optmization problem. The current time-step is defined by the parameters in the
        ``self.opt_p_num`` CasADi structured Data.
        These include the initial condition, the parameters, the time-varying paramters and the previous input.
        Typically, ``self.opt_p_num`` is prepared for the current iteration in the ``.make_step()`` (in MHE/MPC) method.
        It is, however, valid and possible to directly set paramters in ``self.opt_p_num`` before calling ``.solve()``.

        Solve updates the ``opt_x_num``, and ``lam_g_num`` attributes of the class.
        In resetting, ``opt_x_num`` to the current solution, the method implicitly
        enables warmstarting the optimizer for the next iteration, since this vector is always used as the initial guess.

        :raises asssertion: optimizer was not setup yet.

        :return: None
        :rtype: None
        """
        assert self.flags['setup'] == True, 'optimizer was not setup yet. Please call optimizer.setup().'

        r = self.S(x0=self.opt_x_num, lbx=self.lb_opt_x, ubx=self.ub_opt_x,  ubg=self.cons_ub, lbg=self.cons_lb, p=self.opt_p_num)
        # Note: .master accesses the underlying vector of the structure.
        self.opt_x_num.master = r['x']
        self.opt_x_num_unscaled.master = r['x']*self.opt_x_scaling
        self.opt_g_num = r['g']
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.solver_stats = self.S.stats()

        # Calculate values of auxiliary expressions (defined in model)
        self.opt_aux_num.master = self.opt_aux_expression_fun(
                self.opt_x_num,
                self.opt_p_num
            )

    def _setup_discretization(self):
        """Private method that creates the discretization for the optimizer (MHE or MPC).
        Returns the integrator function (``ifcn``) and the total number of collocation points.

        The following discretization methods are available:

        * orthogonal collocation

        * discrete dynamics

        Discretization parameters can be set with the :py:func:`do_mpc.controller.MPC.set_param` and
        :py:func:`do_mpc.estimator.MHE.set_param` methods.

        There is no point in calling this method as part of the public API.
        """
        _x, _u, _z, _tvp, _p = self.model['x', 'u', 'z', 'tvp', 'p']

        rhs = substitute(self.model._rhs, _x, _x*self._x_scaling.cat)
        rhs = substitute(rhs, _u, _u*self._u_scaling.cat)
        rhs = substitute(rhs, _z, _z*self._z_scaling.cat)
        rhs = substitute(rhs, _p, _p*self._p_scaling.cat) # only meaningful for MHE.

        if self.state_discretization == 'discrete':
            _i = SX.sym('i', 0)
            # discrete integrator ifcs mimics the API the collocation ifcn.
            ifcn = Function('ifcn', [_x, _i, _u, _z, _tvp, _p], [[], rhs/self._x_scaling.cat])
            n_total_coll_points = 0
        if self.state_discretization == 'collocation':
            ffcn = Function('ffcn', [_x, _u, _z, _tvp, _p], [rhs/self._x_scaling.cat])
            # Get collocation information
            coll = self.collocation_type
            deg = self.collocation_deg
            ni = self.collocation_ni
            nk = self.n_horizon
            t_step = self.t_step
            n_x = self.model.n_x
            n_u = self.model.n_u
            n_p = self.model.n_p
            n_z = self.model.n_z
            n_tvp = self.model.n_tvp
            n_total_coll_points = (deg + 1) * ni

            # Choose collocation points
            if coll == 'legendre':    # Legendre collocation points
                tau_root = [0] + collocation_points(deg, 'legendre')
            elif coll == 'radau':     # Radau collocation points
                tau_root = [0] + collocation_points(deg, 'radau')
            else:
                raise Exception('Unknown collocation scheme')

            # Size of the finite elements
            h = t_step / ni

            # Coefficients of the collocation equation
            C = np.zeros((deg + 1, deg + 1))

            # Coefficients of the continuity equation
            D = np.zeros(deg + 1)

            # Dimensionless time inside one control interval
            tau = SX.sym("tau")

            # All collocation time points
            T = np.zeros((nk, ni, deg + 1))
            for k in range(nk):
                for i in range(ni):
                    for j in range(deg + 1):
                        T[k, i, j] = h * (k * ni + i + tau_root[j])

            # For all collocation points
            for j in range(deg + 1):
                # Construct Lagrange polynomials to get the polynomial basis at the
                # collocation point
                L = 1
                for r in range(deg + 1):
                    if r != j:
                        L *= (tau - tau_root[r]) / (tau_root[j] - tau_root[r])
                lfcn = Function('lfcn', [tau], [L])
                D[j] = lfcn(1.0)
                # Evaluate the time derivative of the polynomial at all collocation
                # points to get the coefficients of the continuity equation
                tfcn = Function('tfcn', [tau], [tangent(L, tau)])
                for r in range(deg + 1):
                    C[j, r] = tfcn(tau_root[r])

            # Define symbolic variables for collocation
            xk0 = SX.sym("xk0", n_x)
            zk = SX.sym("zk", n_z)
            pk = SX.sym("pk", n_p)
            tv_pk = SX.sym("tv_pk", n_tvp)
            uk = SX.sym("uk", n_u)

            # State trajectory
            n_ik = ni * (deg + 1) * n_x
            ik = SX.sym("ik", n_ik)
            ik_split = np.resize(np.array([], dtype=SX), (ni, deg + 1))
            offset = 0

            # Store initial condition
            ik_split[0, 0] = xk0
            first_j = 1  # Skip allocating x for the first collocation point for the first finite element
            # For each finite element
            for i in range(ni):
                # For each collocation point
                for j in range(first_j, deg + 1):
                    # Get the expression for the state vector
                    ik_split[i, j] = ik[offset:offset + n_x]
                    offset += n_x

                # All collocation points in subsequent finite elements
                first_j = 0

            # Get the state at the end of the control interval
            xkf = ik[offset:offset + n_x]
            offset += n_x
            # Check offset for consistency
            assert(offset == n_ik)

            # Constraints in the control interval
            gk = []
            lbgk = []
            ubgk = []

            # For all finite elements
            for i in range(ni):
                # For all collocation points
                for j in range(1, deg + 1):
                    # Get an expression for the state derivative at the coll point
                    xp_ij = 0
                    for r in range(deg + 1):
                        xp_ij += C[r, j] * ik_split[i, r]

                    # Add collocation equations to the NLP
                    f_ij = ffcn(ik_split[i, j], uk, zk, tv_pk, pk)
                    gk.append(h * f_ij - xp_ij)
                    lbgk.append(np.zeros(n_x))  # equality constraints
                    ubgk.append(np.zeros(n_x))  # equality constraints

                # Get an expression for the state at the end of the finite element
                xf_i = 0
                for r in range(deg + 1):
                    xf_i += D[r] * ik_split[i, r]

                # Add continuity equation to NLP
                x_next = ik_split[i + 1, 0] if i + 1 < ni else xkf
                gk.append(x_next - xf_i)
                lbgk.append(np.zeros(n_x))
                ubgk.append(np.zeros(n_x))

            # Concatenate constraints
            gk = vertcat(*gk)
            lbgk = np.concatenate(lbgk)
            ubgk = np.concatenate(ubgk)

            assert(gk.shape[0] == ik.shape[0])

            # Create the integrator function
            ifcn = Function("ifcn", [xk0, ik, uk, zk, tv_pk, pk], [gk, xkf])

            # Return the integration function and the number of collocation points
        return ifcn, n_total_coll_points

    def _setup_scenario_tree(self):
        """Private method that builds the scenario tree given the possible values of the uncertain parmeters.
        By default all possible combinations of uncertain parameters are evaluated.
        See the API in :py:class:`do_mpc.controller.MHE` for the high level / low level API.
        This method is currently only used for the MPC controller.

        There is no point in calling this method as part of the public API.
        """

        n_p = self.model.n_p
        nk = self.n_horizon
        n_robust = self.n_robust
        # Build auxiliary variables that code the structure of the tree
        # Number of branches
        n_branches = [self.n_combinations if k < n_robust else 1 for k in range(nk)]
        # Calculate the number of scenarios (nodes at each stage)
        n_scenarios = [self.n_combinations**min(k, n_robust) for k in range(nk + 1)]
        # Scenaro tree structure
        child_scenario = -1 * np.ones((nk, n_scenarios[-1], n_branches[0])).astype(int)
        parent_scenario = -1 * np.ones((nk + 1, n_scenarios[-1])).astype(int)
        branch_offset = -1 * np.ones((nk, n_scenarios[-1])).astype(int)
        structure_scenario = np.zeros((nk + 1, n_scenarios[-1])).astype(int)
        # Fill in the auxiliary structures
        for k in range(nk):
            # Scenario counter
            scenario_counter = 0
            # For all scenarios
            for s in range(n_scenarios[k]):
                # For all uncertainty realizations
                for b in range(n_branches[k]):
                    child_scenario[k][s][b] = scenario_counter
                    structure_scenario[k][scenario_counter] = s
                    structure_scenario[k+1][scenario_counter] = s
                    parent_scenario[k + 1][scenario_counter] = s
                    scenario_counter += 1
                # Store the range of branches
                if n_robust == 0:
                    branch_offset[k][s] = 0
                elif k < n_robust:
                    branch_offset[k][s] = 0
                else:
                    branch_offset[k][s] = s % n_branches[0]

        self.scenario_tree = {
            'structure_scenario': structure_scenario,
            'n_branches': n_branches,
            'n_scenarios': n_scenarios,
            'parent_scenario': parent_scenario,
            'branch_offset': branch_offset
        }
        return n_branches, n_scenarios, child_scenario, parent_scenario, branch_offset
