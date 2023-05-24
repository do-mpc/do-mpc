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
Shared tools for optimization-based estimation (MHE) and control (MPC).
"""
import numpy as np
import casadi.tools as castools
import pdb
import do_mpc
from typing import Union,Callable
import os
import subprocess

class Optimizer:
    """The base clase for the optimization based state estimation (MHE) and predictive controller (MPC).
    This class establishes the jointly used attributes, methods and properties.

    Warnings:
        The ``Optimizer`` base class can not be used independently. The methods and properties are 
        inherited to :py:class:`do_mpc.estimator.MHE` and :py:class:`do_mpc.controller.MPC`.
    """
    def __init__(self):
        assert 'model' in self.__dict__.keys(), 'Cannot initialize the optimizer before assigning the model to the current class instance.'

        # Initialize structures for bounds, scaling, initial values by calling the symbolic structures defined in the model
        # with the default numerical value.
        # This returns an identical numerical structure with all values set to the passed value.

        self.flags = {
            'prepare_nlp': False,
            'setup': False,
            'initial_run': False,
        }

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

        # Dummy variables for bounds of all optimization variables
        self._lb_opt_x = None
        self._ub_opt_x = None

        # Lists for further non-linear constraints (optional). Constraints are formulated as cons < ub
        self.nl_cons_list = [
            {'expr_name': 'default', 'expr': castools.DM([]), 'ub': castools.DM([])}
        ]
        self.slack_vars_list = [
            {'slack_name': 'default', 'shape':0, 'ub': castools.DM([]), 'penalty': 0}
        ]
        self.slack_cost = 0

    @property
    def nlp_obj(self):
        """Query and modify (symbolically) the NLP objective function.
        Use the variables in :py:attr:`opt_x` and :py:attr:`opt_p`.

        It is advised to add to the current objective, e.g.:

        ::

            mpc.prepare_nlp()
            # Modify the objective
            mpc.nlp_obj += sum1(vertcat(*mpc.opt_x['_x', -1, 0])**2)
            # Finish creating the NLP
            mpc.create_nlp()

        See the documentation of :py:attr:`opt_x` and :py:attr:`opt_p` on how to query these attributes.

        Warnings:
            This is a VERY low level feature and should be used with extreme caution.
            It is easy to break the code.

            Be especially careful NOT to accidentially overwrite the default objective.

        Note:
            Modifications must be done after calling :py:meth:`prepare_nlp`
            and before calling :py:meth:`create_nlp`
        """
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling MPC.prepare_nlp or MPC.setup'
        return self._nlp_obj

    @nlp_obj.setter
    def nlp_obj(self, val):
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling MPC.prepare_nlp or MPC.setup'
        assert not self.flags['setup'], 'Cannot change attribute after calling MPC.create_nlp or MPC.setup'
        self._nlp_obj = val

    @property
    def nlp_cons(self):
        """Query and modify (symbolically) the NLP constraints.
        Use the variables in :py:attr:`opt_x` and :py:attr:`opt_p`.

        Prior to calling :py:meth:`create_nlp` this attribute returns a list of symbolic constraints.
        After calling :py:meth:`create_nlp` this attribute returns the concatenation of this list
        and the attribute cannot be altered anymore.

        It is advised to append to the current list of :py:attr:`nlp_cons`:

        ::

            mpc.prepare_nlp()

            # Create new constraint: Input at timestep 0 and 1 must be identical.
            extra_cons = mpc.opt_x['_u', 0, 0]-mpc.opt_x['_u',1, 0]
            mpc.nlp_cons.append(
                extra_cons
            )

            # Create appropriate upper and lower bound (here they are both 0 to create an equality constraint)
            mpc.nlp_cons_lb.append(np.zeros(extra_cons.shape))
            mpc.nlp_cons_ub.append(np.zeros(extra_cons.shape))

            mpc.create_nlp()

        See the documentation of :py:attr:`opt_x` and :py:attr:`opt_p` on how to query these attributes.

        Warnings:
            This is a VERY low level feature and should be used with extreme caution.
            It is easy to break the code.

            Be especially careful NOT to accidentially overwrite the default objective.

        Note:
            Modifications must be done after calling :py:meth:`prepare_nlp`
            and before calling :py:meth:`create_nlp`
        """
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling MPC.prepare_nlp or MPC.setup'
        return self._nlp_cons

    @nlp_cons.setter
    def nlp_cons(self, val):
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling prepare_nlp or setup'
        assert not self.flags['setup'], 'Cannot change attribute after calling create_nlp or setup'
        self._nlp_cons = val


    @property
    def nlp_cons_lb(self):
        """Query and modify the lower bounds of the :py:attr:`nlp_cons`.

        Prior to calling :py:meth:`create_nlp` this attribute returns a list of lower bounds
        matching the list of constraints obtained with :py:attr:`nlp_cons`.
        After calling :py:meth:`create_nlp` this attribute returns the concatenation of this list.

        Values for lower (and upper) bounds MUST be added when adding new constraints to :py:attr:`nlp_cons`.

        Warnings:
            This is a VERY low level feature and should be used with extreme caution.
            It is easy to break the code.

        Note:
            Modifications must be done after calling :py:meth:`prepare_nlp`
        """
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling MPC.prepare_nlp or MPC.setup'
        return self._nlp_cons_lb

    @nlp_cons_lb.setter
    def nlp_cons_lb(self, val):
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling prepare_nlp or setup'
        self._nlp_cons_lb = val

    @property
    def nlp_cons_ub(self):
        """Query and modify the upper bounds of the :py:attr:`nlp_cons`.

        Prior to calling :py:meth:`create_nlp` this attribute returns a list of upper bounds
        matching the list of constraints obtained with :py:attr:`nlp_cons`.
        After calling :py:meth:`create_nlp` this attribute returns the concatenation of this list.

        Values for upper (and lower) bounds MUST be added when adding new constraints to :py:attr:`nlp_cons`.

        Warnings:
            This is a VERY low level feature and should be used with extreme caution.
            It is easy to break the code.

        Note:
            Modifications must be done after calling :py:meth:`prepare_nlp`
        """
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling MPC.prepare_nlp or MPC.setup'
        return self._nlp_cons_ub

    @nlp_cons_ub.setter
    def nlp_cons_ub(self, val):
        assert self.flags['prepare_nlp'], 'Cannot query attribute prior to calling prepare_nlp or setup'
        self._nlp_cons_ub = val

    @do_mpc.tools.IndexedProperty
    def lb_opt_x(self, ind):
        """Query and modify the lower bounds of all optimization variables :py:attr:`opt_x`.
        This is a more advanced method of setting bounds on optimization variables of the MPC/MHE problem.
        Users with less experience are advised to use :py:attr:`bounds` instead.

        The attribute returns a nested structure that can be indexed using powerindexing. Please refer to :py:attr:`opt_x` for more details. 

        Note:
            The attribute automatically considers the scaling variables when setting the bounds. See :py:attr:`scaling` for more details.

        Note:
            Modifications must be done after calling :py:meth:`prepare_nlp` or :py:meth:`setup` respectively.
        """
        return self._lb_opt_x[ind] 

    @lb_opt_x.setter
    def lb_opt_x(self, ind, val):
        self._lb_opt_x[ind] = val
        # Get canonical index 
        cind = self._lb_opt_x.f[ind]
        # Modify the newly set values by considering the scaling variables. This requires the canonical index.
        self._lb_opt_x.master[cind] = self._lb_opt_x.master[cind]/self.opt_x_scaling.master[cind]
        


    @do_mpc.tools.IndexedProperty
    def ub_opt_x(self, ind):
        """Query and modify the lower bounds of all optimization variables :py:attr:`opt_x`.
        This is a more advanced method of setting bounds on optimization variables of the MPC/MHE problem.
        Users with less experience are advised to use :py:attr:`bounds` instead.

        The attribute returns a nested structure that can be indexed using powerindexing. Please refer to :py:attr:`opt_x` for more details. 

        Note:
            The attribute automatically considers the scaling variables when setting the bounds. See :py:attr:`scaling` for more details.

        Note:
            Modifications must be done after calling :py:meth:`prepare_nlp` or :py:meth:`setup` respectively.
        """
        return self._ub_opt_x[ind]

    @ub_opt_x.setter
    def ub_opt_x(self, ind, val):
        self._ub_opt_x[ind] = val
        # Get canonical index
        cind = self._ub_opt_x.f[ind]
        # Modify the newly set values by considering the scaling variables. This requires the canonical index.
        self._ub_opt_x.master[cind] = self._ub_opt_x.master[cind]/self.opt_x_scaling.master[cind]


    @do_mpc.tools.IndexedProperty
    def bounds(self, ind):
        """Query and set bounds of the optimization variables.
        The :py:func:`bounds` method is an indexed property, meaning
        getting and setting this property requires an index and calls this function.
        The power index (elements are separated by commas) must contain atleast the following elements:

        ======      =================   ==========================================================
        order       index name          valid options
        ======      =================   ==========================================================
        1           bound type          ``lower`` and ``upper``
        2           variable type       ``_x``, ``_u`` and ``_z`` (and ``_p_est`` for MHE)
        3           variable name       Names defined in :py:class:`do_mpc.model.Model`.
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
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

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
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

        # Set value on struct:
        var_struct[var_name] = val

        # Update bounds of optimization variables, if the problem is already created:
        if self.flags['prepare_nlp']:
            self._update_bounds()


    @do_mpc.tools.IndexedProperty
    def scaling(self, ind):
        """Query and set  scaling of the optimization variables.
        The :py:func:`Optimizer.scaling` method is an indexed property, meaning
        getting and setting this property requires an index and calls this function.
        The power index (elements are seperated by comas) must contain atleast the following elements:

        ======      =================   ==========================================================
        order       index name          valid options
        ======      =================   ==========================================================
        1           variable type       ``_x``, ``_u`` and ``_z`` (and ``_p_est`` for MHE)
        2           variable name       Names defined in :py:class:`do_mpc.model.Model`.
        ======      =================   ==========================================================

        Further indices are possible (but not neccessary) when the referenced variable is a vector or matrix.


        **Example**:

        ::

            # Set with:
            optimizer.scaling['_x', 'phi_1'] = 2
            optimizer.scaling['_x', 'phi_2'] = 2

            # Query with:
            optimizer.scaling['_x', 'phi_1']

        Scaling factors :math:`a` affect the MHE / MPC optimization problem. The optimization variables are scaled variables:

        .. math::

            \\bar\\phi = \\frac{\\phi}{a_{\\phi}} \\quad \\forall \\phi \\in [x, u, z, p_{\\text{est}}]

        Scaled variables are used to formulate the bounds :math:`\\bar\\phi_{lb} \\leq \\bar\\phi_{ub}`
        and for the evaluation of the ODE. For the objective function and the nonlinear constraints
        the unscaled variables are used. The algebraic equations are also not scaled.

        Note:
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
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

        return var_struct[var_name]


    @scaling.setter
    def scaling(self, ind, val):
        """See Docstring for scaling getter method"""
        assert not self.flags['setup'], 'Scaling can only be set before the optimization problem is created.'
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
        assert (var_name[0] if isinstance(var_name, tuple) else var_name) in var_struct.keys(), err_msg.format(ind, var_struct.keys())

        var_struct[var_name] = val

    def reset_history(self)->None:
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

        if self.settings.store_full_solution == True:
            # Create data_field for the optimal solution.
            self.data.data_fields.update({'_opt_x_num': self.n_opt_x})
            self.data.data_fields.update({'_opt_aux_num': self.n_opt_aux})
            self.data.opt_x = self.opt_x
            # aux_struct is the struct_symSX variant of opt_aux (which is struct_SX). struct_SX cannot be unpickled (bug).
            # See: https://groups.google.com/forum/#!topic/casadi-users/dqAb4tnA2ik
            self.data.opt_aux = self.aux_struct
        if self.settings.store_lagr_multiplier == True:
            # Create data_field for the lagrange multipliers
            self.data.data_fields.update({'_lam_g_num': self.n_opt_lagr})
        if len(self.settings.store_solver_stats) > 0:
            # These are valid arguments for solver stats:
            solver_stats = ['iter_count', 'iterations', 'n_call_S', 'n_call_callback_fun',
                            'n_call_nlp_f', 'n_call_nlp_g', 'n_call_nlp_grad', 'n_call_nlp_grad_f',
                            'n_call_nlp_hess_l', 'n_call_nlp_jac_g', 'return_status', 'success', 't_proc_S',
                            't_proc_callback_fun', 't_proc_nlp_f', 't_proc_nlp_g', 't_proc_nlp_grad',
                            't_proc_nlp_grad_f', 't_proc_nlp_hess_l', 't_proc_nlp_jac_g', 't_wall_total',
                            't_wall_callback_fun', 't_wall_nlp_f', 't_wall_nlp_g', 't_wall_nlp_grad', 't_wall_nlp_grad_f',
                            't_wall_nlp_hess_l', 't_wall_nlp_jac_g']
            # Create data_field(s) for the recorded (valid) stats.
            for stat_i in self.settings.store_solver_stats:
                assert stat_i in solver_stats, 'The requested {} is not a valid solver stat and cannot be recorded. Please supply one of the following (or none): {}'.format(stat_i, solver_stats)
                self.data.data_fields.update({stat_i: 1})

        self.data.init_storage()

    def set_nl_cons(self, 
                    expr_name:str, 
                    expr:Union[castools.SX,castools.MX], 
                    ub:float=np.inf, 
                    soft_constraint:bool=False, 
                    penalty_term_cons:int=1, 
                    maximum_violation:float=np.inf)->Union[castools.SX,castools.MX]:
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

        Args:
            expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
            expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
            ub: Upper bound
            soft_constraint: Flag to enable soft constraint
            penalty_term_cons: Penalty term constant
            maximum_violation: Maximum violation

        Raises:
            assertion: expr_name must be str
            assertion: expr must be a casadi SX or MX type

        Returns:
            Returns the newly created expression. Expression can be used e.g. for the RHS.
        """
        assert self.flags['setup'] == False, 'Cannot call .set_expression after .setup().'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (castools.SX, castools.MX)), 'expr must be a casadi SX or MX type, you have: {}'.format(type(expr))
        assert isinstance(ub, (int, float, np.ndarray)), 'ub must be float, int or numpy.ndarray, you have: {}'.format(type(ub))
        assert isinstance(soft_constraint, bool), 'soft_constraint must be boolean, you have: {}'.format(type(soft_constraint))

        if soft_constraint==True:
            # Add slack variable to list of slack variables:
            self.slack_vars_list.extend([
                {'slack_name': expr_name, 'shape': expr.shape, 'ub': maximum_violation, 'penalty': penalty_term_cons}
            ])

        # All operations to make the soft constraints work are in _setup_nl_cons.

        self.nl_cons_list.extend([
            {'expr_name': expr_name, 'expr': expr, 'ub' : ub}])

        return expr

    def _setup_nl_cons(self, nl_cons_input:Union[castools.SX,castools.MX])->None:
        """Private method that is called from :py:func:`do_mpc.controller.MPC.setup` or :py:func:`do_mpc.estimator.MHE.setup`.
        Afterwards no further non-linear constraints can be added with the :py:func:`Optimizer.set_nl_cons` method.

        This is not part of the public API. Do not call this method.

        Args:
            nl_cons_input: list of symbolic variables used as input to the nl_cons function.
        """
        # Create struct for soft constraints:
        self._eps = _eps = self.model.sv.sym_struct([
            castools.entry(slack_i['slack_name'], shape=slack_i['shape']) for slack_i in self.slack_vars_list
        ])
        # Create struct for _nl_cons:
        # Use the previously defined sym variables to declare shape and symbolic variable.
        self._nl_cons = self.model.sv.struct([
            castools.entry(expr_i['expr_name'], expr=expr_i['expr']) for expr_i in self.nl_cons_list
        ])

        self.n_eps = _eps.shape[0]
        # Create bounds:
        self._eps_lb = _eps(0.0)
        self._eps_ub = _eps(np.inf)

        # Set bounds, add slack variable to constraint and add cost.
        for slack_i in self.slack_vars_list:
            self._eps_ub[slack_i['slack_name']] = slack_i['ub']
            self._nl_cons[slack_i['slack_name']] -= self._eps[slack_i['slack_name']]
            self.slack_cost += castools.sum1(slack_i['penalty']*self._eps[slack_i['slack_name']])

        # Objective function epsilon contribution:
        self.epsterm_fun = castools.Function('epsterm', [_eps], [self.slack_cost])

        # Make function from these expressions:
        nl_cons_input += [_eps]
        self._nl_cons_fun = castools.Function('nl_cons_fun', nl_cons_input, [self._nl_cons])

        # Create bounds:
        self._nl_cons_ub = self._nl_cons(np.inf)
        self._nl_cons_lb = self._nl_cons(-np.inf)
        # Set bounds:
        for nl_cons_i in self.nl_cons_list:
            self._nl_cons_ub[nl_cons_i['expr_name']] = nl_cons_i['ub']


    def get_tvp_template(self)->Union[castools.structure3.SXStruct,castools.structure3.MXStruct]:
        """Obtain output template for :py:func:`set_tvp_fun`.

        The method returns a structured object with ``n_horizon+1`` elements,
        and a set of time-varying parameters (as defined in :py:class:`do_mpc.model.Model`)
        for each of these instances. The structure is initialized with all zeros.
        Use this object to define values of the time-varying parameters.

        This structure (with numerical values) should be used as the output of the ``tvp_fun`` function which is set to the class with :py:func:`set_tvp_fun`.
        Use the combination of :py:func:`get_tvp_template` and :py:func:`set_tvp_fun`.

        **Example:**

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

        Returns:
            Casadi SX or MX structure
        """

        tvp_template = self.model.sv.sym_struct([
            castools.entry('_tvp', repeat=self.settings.n_horizon+1, struct=self.model._tvp)
        ])
        return tvp_template(0)

    def set_tvp_fun(self, tvp_fun:Callable[[float],Union[castools.structure3.SXStruct,castools.structure3.MXStruct]])->None:
        """ Set function which returns time-varying parameters.

        The ``tvp_fun`` is called at each optimization step to get the current prediction of the time-varying parameters.
        The supplied function must be callable with the current time as the only input. Furthermore, the function must return
        a CasADi structured object which is based on the horizon and on the model definition. The structure can be obtained with
        :py:func:`get_tvp_template`.

        **Example:**

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

        Note:
            The method :py:func:`set_tvp_fun`. must be called prior to setup IF time-varying parameters are defined in the model.
            It is not required to call the method if no time-varying parameters are defined.

        Args:
            tvp_fun: Function that returns the predicted tvp values at each timestep. Must have single input (float) and return a ``structure3.DMStruct`` (obtained with :py:func:`get_tvp_template`).
        """
        assert isinstance(tvp_fun(0), castools.structure3.DMStruct), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'

        self.flags['set_tvp_fun'] = True

        self.tvp_fun = tvp_fun

    def compile_nlp(self, overwrite:bool = False, cname:str = 'nlp.c', libname:str='nlp.so', compiler_command:str=None)->None:
        """Compile the NLP. This may accelerate the optimization.
        As compilation is time consuming, the default option is to NOT overwrite (``overwrite=False``) an existing compilation.
        If an existing compilation with the name ``libname`` is found, it is used. **This can be dangerous, if the NLP has changed**
        (user tweaked the cost function, the model etc.).

        Warnings: 
            This feature is experimental and currently only supported on Linux and MacOS.

        **What happens here?**
        
        1. The NLP is written to a C-file (``cname``)
        
        2. The C-File (``cname``) is compiled. The custom compiler uses:

        ::

            gcc -fPIC -shared -O1 {cname} -o {libname}

        3. The compiled library is linked to the NLP. This overwrites the original NLP. Options from the previous NLP (e.g. linear solver) are kept.

        ::

            self.S = nlpsol('solver_compiled', 'ipopt', f'{libname}', self.nlpsol_opts)      

        Args:
            overwrite: If True, the existing compiled NLP will be overwritten.
            cname: Name of the C file that will be exported.
            libname: Name of the shared library that will be created after compilation.
            compiler_command: Command to use for compiling. If None, the default compiler command will be used. Please make sure to use matching strings for ``libname`` when supplying your custom compiler command.
        """
        if not self.flags['setup']:
            raise Exception('Optimizer not setup. Call setup first.')

        if castools.sys.platform  not in ('darwin', 'linux', 'linux2'):
            raise Exception('Compilation not supported on this platform.')

        if compiler_command is None:
            compiler_command = "gcc -fPIC -shared -O1 {cname} -o {libname}".format(cname=cname, libname=libname)

        # Only compile if not already compiled:
        if overwrite or not os.path.isfile(libname):
            # Create c code from solver object
            print('Generating c-code of nlp.')
            self.S.generate_dependencies(cname)
            # Compile c code
            print('Compiling c-code of nlp.')
            subprocess.Popen(compiler_command, shell=True).wait()

        # Overwrite solver object with loaded nlp:
        self.S = nlpsol('solver_compiled', 'ipopt', libname, self.settings.nlpsol_opts)
        print('Using compiled NLP solver.')

    def solve(self)->None:
        """Solves the optmization problem.

        The current problem is defined by the parameters in the
        :py:attr:`opt_p_num` CasADi structured Data.

        Typically, :py:attr:`opt_p_num` is prepared for the current iteration in the :py:func:`make_step` method.
        It is, however, valid and possible to directly set paramters in :py:attr:`opt_p_num` before calling :py:func:`solve`.

        The method updates the :py:attr:`opt_p_num` and :py:attr:`opt_x_num` attributes of the class.
        By resetting :py:attr:`opt_x_num` to the current solution, the method implicitly
        enables **warmstarting the optimizer** for the next iteration, since this vector is always used as the initial guess.

        Warnings:
            The method is part of the public API but it is generally not advised to use it.
            Instead we recommend to call :py:func:`make_step` at each iterations, which acts as a wrapper
            for :py:func:`solve`.

        Raises:
            asssertion: Optimizer was not setup yet.
        """
        assert self.flags['setup'] == True, 'optimizer was not setup yet. Please call optimizer.setup().'

        solver_call_kwargs = {
            'x0': self.opt_x_num,
            'lbx': self._lb_opt_x,
            'ubx': self._ub_opt_x,
            'lbg': self.nlp_cons_lb,
            'ubg': self.nlp_cons_ub,
            'p': self.opt_p_num,
        }

        # Warmstarting the optimizer after the initial run:
        if self.flags['initial_run']:
            solver_call_kwargs.update({
                'lam_x0': self.lam_x_num,
                'lam_g0': self.lam_g_num,
            })

        r = self.S(**solver_call_kwargs)
        # Note: .master accesses the underlying vector of the structure.
        self.opt_x_num.master = r['x']
        self.opt_x_num_unscaled.master = r['x']*self.opt_x_scaling
        self.opt_g_num = r['g']
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.lam_x_num = r['lam_x']
        self.solver_stats = self.S.stats()

        # Calculate values of auxiliary expressions (defined in model)
        self.opt_aux_num.master = self.opt_aux_expression_fun(
                self.opt_x_num,
                self.opt_p_num
            )

        # For warmstarting purposes: Flag that initial run has been completed.
        self.flags['initial_run'] = True

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
        # Scaled variables
        _x, _u, _z, _tvp, _p, _w = self.model['x', 'u', 'z', 'tvp', 'p', 'w']

        # Unscale variables
        _x_unscaled = _x*self._x_scaling.cat
        _u_unscaled = _u*self._u_scaling.cat
        _z_unscaled = _z*self._z_scaling.cat
        _p_unscaled = _p*self._p_scaling.cat

        # Create _rhs and _alg
        _rhs = self.model._rhs_fun(_x_unscaled, _u_unscaled, _z_unscaled, _tvp, _p_unscaled, _w)
        _alg = self.model._alg_fun(_x_unscaled, _u_unscaled, _z_unscaled, _tvp, _p_unscaled, _w)

        # Scale (only _rhs)
        _rhs_scaled = _rhs/self._x_scaling.cat

        if self.model.model_type == 'discrete':
            _i = self.model.sv.sym('i', 0)
            # discrete integrator ifcs mimics the API the collocation ifcn.
            ifcn = castools.Function('ifcn', [_x, _i, _u, _z, _tvp, _p, _w], [_alg, _rhs_scaled])
            n_total_coll_points = 0
        elif self.settings.state_discretization == 'collocation':
            ffcn = castools.Function('ffcn', [_x, _u, _z, _tvp, _p, _w], [_rhs_scaled])
            afcn = castools.Function('afcn', [_x, _u, _z, _tvp, _p, _w], [_alg])
            # Get collocation information
            coll = self.settings.collocation_type
            deg = self.settings.collocation_deg
            ni = self.settings.collocation_ni
            nk = self.settings.n_horizon
            t_step = self.settings.t_step
            n_x = self.model.n_x
            n_u = self.model.n_u
            n_p = self.model.n_p
            n_z = self.model.n_z
            n_w = self.model.n_w
            n_tvp = self.model.n_tvp
            n_total_coll_points = (deg + 1) * ni

            # Choose collocation points
            if coll == 'legendre':    # Legendre collocation points
                tau_root = [0] + castools.collocation_points(deg, 'legendre')
            elif coll == 'radau':     # Radau collocation points
                tau_root = [0] + castools.collocation_points(deg, 'radau')
            else:
                raise Exception('Unknown collocation scheme')

            # Size of the finite elements
            h = t_step / ni

            # Coefficients of the collocation equation
            C = np.zeros((deg + 1, deg + 1))

            # Coefficients of the continuity equation
            D = np.zeros(deg + 1)

            # Dimensionless time inside one control interval
            tau = self.model.sv.sym("tau")

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
                lfcn = castools.Function('lfcn', [tau], [L])
                D[j] = lfcn(1.0)
                # Evaluate the time derivative of the polynomial at all collocation
                # points to get the coefficients of the continuity equation
                tfcn = castools.Function('tfcn', [tau], [castools.tangent(L, tau)])
                for r in range(deg + 1):
                    C[j, r] = tfcn(tau_root[r])

            # Define symbolic variables for collocation
            xk0 = self.model.sv.sym("xk0", n_x)
            #zk = self.model.sv.sym("zk", n_z)
            pk = self.model.sv.sym("pk", n_p)
            tv_pk = self.model.sv.sym("tv_pk", n_tvp)
            uk = self.model.sv.sym("uk", n_u)
            wk = self.model.sv.sym("wk", n_w)

            # State trajectory
            n_ik = ni * (deg + 1) * n_x
            ik = self.model.sv.sym("ik", n_ik)

            ik_split = np.resize(np.array([], dtype=self.model.sv.dtype), (ni, deg + 1))
            offset = 0

            # Algebraic trajectory
            n_zk = ni * (deg +1) * n_z
            zk = self.model.sv.sym("zk", n_zk)
            offset_z = 0
            zk_split = np.resize(np.array([], dtype=self.model.sv.dtype), (ni, deg + 1))

            # Store initial condition
            ik_split[0, 0] = xk0
            zk_split[0, 0] = zk[offset_z:offset_z + n_z]
            offset_z += n_z
            first_j = 1  # Skip allocating x for the first collocation point for the first finite element
            # For each finite element
            for i in range(ni):
                # For each collocation point
                for j in range(first_j, deg + 1):
                    # Get the expression for the state vector
                    ik_split[i, j] = ik[offset:offset + n_x]
                    zk_split[i, j] = zk[offset_z:offset_z + n_z]
                    offset_z += n_z
                    offset += n_x

                # All collocation points in subsequent finite elements
                first_j = 0

            # Get the state at the end of the control interval
            xkf = ik[offset:offset + n_x]
            offset += n_x
            # Check offset for consistency
            assert(offset == n_ik)
            assert(offset_z == n_zk)
            # Constraints in the control interval
            gk = []
            lbgk = []
            ubgk = []

            # For all finite elements
            for i in range(ni):
                # for the first point:
                a_i0 = afcn(ik_split[i, 0], uk, zk_split[i,0], tv_pk, pk, wk)
                gk.append(a_i0)
                lbgk.append(np.zeros(n_z))
                ubgk.append(np.zeros(n_z))

                # For all collocation points
                for j in range(1, deg + 1):
                    # Get an expression for the state derivative at the coll point
                    xp_ij = 0
                    for r in range(deg + 1):
                        xp_ij += C[r, j] * ik_split[i, r]

                    # Add collocation equations to the NLP
                    f_ij = ffcn(ik_split[i, j], uk, zk_split[i,j], tv_pk, pk, wk)
                    gk.append(h * f_ij - xp_ij)
                    lbgk.append(np.zeros(n_x))  # equality constraints
                    ubgk.append(np.zeros(n_x))  # equality constraints

                    # algebraic constraints
                    a_ij = afcn(ik_split[i, j], uk, zk_split[i,j], tv_pk, pk, wk)
                    gk.append(a_ij)
                    lbgk.append(np.zeros(n_z))
                    ubgk.append(np.zeros(n_z))


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
            gk = castools.vertcat(*gk)
            lbgk = np.concatenate(lbgk)
            ubgk = np.concatenate(ubgk)

            assert(gk.shape[0] == ik.shape[0] + zk.shape[0])

            # Create the integrator function
            ifcn = castools.Function("ifcn", [xk0, ik, uk, zk, tv_pk, pk, wk], [gk, xkf])

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
        nk = self.settings.n_horizon
        n_robust = self.settings.n_robust
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

    def prepare_nlp(self)->None:
        """Prepare the optimization problem.
        Typically, this method is called internally from :py:meth:`setup`.

        Users should only call this method if they intend to modify the objective with :py:attr:`nlp_obj`,
        the constraints with :py:attr:`nlp_cons`, :py:attr:`nlp_cons_lb` and :py:attr:`nlp_cons_ub`.

        To finish the setup process, users MUST call :py:meth:`create_nlp` afterwards.

        Note:
            Do NOT call :py:meth:`setup` if you intend to go the manual route with :py:meth:`prepare_nlp` and :py:meth:`create_nlp`.

        Note:
            Only AFTER calling :py:meth:`prepare_nlp` the previously mentionned attributes
            :py:attr:`nlp_obj`, :py:attr:`nlp_cons`, :py:attr:`nlp_cons_lb`, :py:attr:`nlp_cons_ub`
            become available.
        
        Returns:
            None
        """
        # MPC and MHE have similar methods. The documentation is valid for both of them.
        self._prepare_nlp()

    def create_nlp(self)->None:
        """Create the optimization problem.
        Typically, this method is called internally from :py:meth:`setup`.

        Users should only call this method if they intend to modify the objective with :py:attr:`nlp_obj`,
        the constraints with :py:attr:`nlp_cons`, :py:attr:`nlp_cons_lb` and :py:attr:`nlp_cons_ub`.

        To finish the setup process, users MUST call :py:meth:`create_nlp` afterwards.

        Note:
            Do NOT call :py:meth:`setup` if you intend to go the manual route with :py:meth:`prepare_nlp` and :py:meth:`create_nlp`.

        Note:
            Only AFTER calling :py:meth:`prepare_nlp` the previously mentionned attributes
            :py:attr:`nlp_obj`, :py:attr:`nlp_cons`, :py:attr:`nlp_cons_lb`, :py:attr:`nlp_cons_ub`
            become available.

        Returns:
            None
        """
        # MPC and MHE have similar methods. The documentation is valid for both of them.
        self._create_nlp()