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


class Model:
    """The **do-mpc** model class. This class holds the full model description and is at the core of
    :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC` and :py:class:`do_mpc.estimator.Estimator`.
    The :py:class:`Model` class is created with setting the ``model_type`` (continuous or discrete).
    A ``continous`` model consists of an underlying ordinary differential equation (ODE) or differential algebraic equation (DAE):

    .. math::

       \dot{x} &= f(x,u,z,p_{tv},p)\\\\
       0 &= g(x,u,z,p_{tv},p)\\\\
       y &= h(x,u,z)

    whereas a ``discrete`` model consists of a difference equation:

    .. math::

       x_{k+1} &= f(x_{k},u_{k},z_{k},p_{tv,k},p)\\\\
       0 &= g(x_{k},u_{k}, z_{k},p_{tv,k},p)\\\\
       y_k &= h(x_k,u_k,z_k)


    **Configuration and setup:**

    Configuring and setting up the optimizer involves the following steps:

    1. Use :py:func:`set_variable` to introduce new variables to the model. It is important to declare the variable type (states, inputs, etc.).

    2. Optionally introduce "auxiliary" expressions as functions of the previously defined variables with :py:func:`set_expression`. The expressions can be used for monitoring or be reused as constraints, the cost function etc.

    3. Optionally introduce measurement equations with :py:func:`set_meas`. The syntax is identical to :py:func:`set_expression`. By default state-feedback is assumed.

    4. Define the right-hand-side of the `discrete` or `continuous` model as a function of the previously defined variables with :py:func:`set_rhs`. This method must be called once for each introduced state.

    5. Call :py:func:`setup` to finalize the :py:class:`Model`. No further changes are possible afterwards.

    :param model_type: Set if the model is ``discrete`` or ``continuous``.
    :type var_type: str

    :raises assertion: model_type must be string
    :raises assertion: model_type must be either discrete or continuous

    .. automethod:: __getitem__
    """
    # Define private class attributes

    _x = []

    _u =   [entry('default', shape=(0,0))]

    _z =   [entry('default', shape=(0,0))]

    _p =   [entry('default', shape=(0,0))]

    _tvp = [entry('default', shape=(0,0))]

    _aux = [entry('default', shape=(0,0))]
    _aux_expression = [entry('default', expr=DM(0))]

    _y =   [entry('default', shape=(0,0))]
    _y_expression = []




    def __init__(self, model_type=None):
        assert isinstance(model_type, str), 'model_type must be string, you have: {}'.format(type(model_type))
        assert model_type in ['discrete', 'continuous'], 'model_type must be either discrete or continuous, you have: {}'.format(model_type)

        self.model_type = model_type

        self.rhs_list = []

        self.flags = {
            'setup': False
        }

    def __getitem__(self, ind):
        """The :py:class:`Model` class supports the ``__getitem__`` method,
        which can be used to retrieve the model variables (see attribute list).

        ::

            # Query the states like this:
            x = model.x
            # or like this:
            x = model['x']

        This also allows to retrieve multiple variables simultaneously:

        ::

            x, u, z = model['x','u','z']
        """
        var_names = ['x','u','z','p','tvp','y','aux']
        if isinstance(ind, tuple):
            val = []
            for ind_i in ind:
                assert ind_i in var_names, 'The queried variable {} is not valid. Choose from {}.'.format(ind_i, var_names)
                val.append(getattr(self, ind_i))
            return val
        else:
            val = getattr(self,ind)

        return val


    @property
    def x(self):
        """ Dynamic states.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            .. note ::

                Variables are introduced with :py:func:`Model.set_variable` Use this property only to query
                variables.

            **Example:**

            ::

                model = do_mpc.model.Model('continuous')
                model.set_variable('_x','temperature', shape=(4,1))
                # Query:
                model.x['temperature', 0] # 0th element of variable
                model.x['temperature']    # all elements of variable
                model.x['temperature', 0:2]    # 0th and 1st element

            Usefull CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``


            :raises assertion: Cannot set model variables direcly. Use set_variable instead.
        """
        if self.flags['setup']:
            return self._x
        else:
            return struct_symSX(self._x)
    @x.setter
    def x(self, val):
        raise Exception('Cannot set model variables direcly. Use set_variable instead.')

    @property
    def u(self):
        """ Inputs.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            .. note ::

                Variables are introduced with :py:func:`Model.set_variable` Use this property only to query
                variables.

            **Example:**

            ::

                model = do_mpc.model.Model('continuous')
                model.set_variable('_u','heating', shape=(4,1))
                # Query:
                model.u['heating', 0] # 0th element of variable
                model.u['heating']    # all elements of variable
                model.u['heating', 0:2]    # 0th and 1st element

            Usefull CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set model variables direcly. Use set_variable instead.
        """
        if self.flags['setup']:
            return self._u
        else:
            return struct_symSX(self._u)
    @u.setter
    def u(self, val):
        raise Exception('Cannot set model variables direcly. Use set_variable instead.')

    @property
    def z(self):
        """ Algebraic states.
        CasADi symbolic structure, can be indexed with user-defined variable names.

        .. note ::

            Variables are introduced with :py:func:`Model.set_variable` Use this property only to query
            variables.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_z','temperature', shape=(4,1))
            # Query:
            model.z['temperature', 0] # 0th element of variable
            model.z['temperature']    # all elements of variable
            model.z['temperature', 0:2]    # 0th and 1st element

        Usefull CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``


        :raises assertion: Cannot set model variables direcly. Use set_variable instead.
        """
        if self.flags['setup']:
            return self._z
        else:
            return struct_symSX(self._z)
    @z.setter
    def z(self, val):
        raise Exception('Cannot set model variables direcly. Use set_variable instead.')

    @property
    def p(self):
        """ Static parameters.
        CasADi symbolic structure, can be indexed with user-defined variable names.

        .. note ::

            Variables are introduced with :py:func:`Model.set_variable` Use this property only to query
            variables.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_p','temperature', shape=(4,1))
            # Query:
            model.p['temperature', 0] # 0th element of variable
            model.p['temperature']    # all elements of variable
            model.p['temperature', 0:2]    # 0th and 1st element

        Usefull CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``


        :raises assertion: Cannot set model variables direcly. Use set_variable instead.
        """
        if self.flags['setup']:
            return self._p
        else:
            return struct_symSX(self._p)
    @p.setter
    def p(self, val):
        raise Exception('Cannot set model variables direcly. Use set_variable instead.')

    @property
    def tvp(self):
        """ Time-varying parameters.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            .. note ::

                Variables are introduced with :py:func:`Model.set_variable` Use this property only to query
                variables.

            **Example:**

            ::

                model = do_mpc.model.Model('continuous')
                model.set_variable('_tvp','temperature', shape=(4,1))
                # Query:
                model.tvp['temperature', 0] # 0th element of variable
                model.tvp['temperature']    # all elements of variable
                model.tvp['temperature', 0:2]    # 0th and 1st element

            Usefull CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set model variables direcly. Use set_variable instead.
        """
        if self.flags['setup']:
            return self._tvp
        else:
            return struct_symSX(self._tvp)
    @tvp.setter
    def tvp(self, val):
        raise Exception('Cannot set model variables direcly. Use set_variable instead.')

    @property
    def y(self):
        """ Measurements.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            .. note ::

                Measured variables are introduced with :py:func:`Model.set_meas` Use this property only to query
                variables.

            **Example:**

            ::

                model = do_mpc.model.Model('continuous')
                model.set_variable('_x','temperature', 4) # 4 states
                model.set_meas('temperature', model.x['temperature',:2]) # first 2 measured
                # Query:
                model.y['temperature', 0] # 0th element of variable
                model.y['temperature']    # all elements of variable

            Usefull CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set model variables direcly. Use set_meas instead.
        """
        if self.flags['setup']:
            return self._y_expression
        else:
            return struct_SX(self._y_expression)
    @y.setter
    def y(self, val):
        raise Exception('Cannot set model variables direcly. Use set_variable instead.')

    @property
    def aux(self):
        """ Auxiliary expressions.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            .. note ::

                Expressions are introduced with :py:func:`Model.set_expression` Use this property only to query
                variables.

            **Example:**

            ::

                model = do_mpc.model.Model('continuous')
                model.set_variable('_x','temperature', 4) # 4 states
                dt = model.x['temperature',0]- model.x['temperature', 1]
                model.set_expression('dtemp', dt)
                # Query:
                model.y['dtemp', 0] # 0th element of variable
                model.y['dtemp']    # all elements of variable

            Usefull CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set aux direcly. Use set_expression instead.
        """
        if self.flags['setup']:
            return self._aux_expression
        else:
            return struct_SX(self._aux_expression)
    @aux.setter
    def aux(self, val):
        raise Exception('Cannot set model variables direcly. Use set_variable instead.')


    def set_variable(self, var_type, var_name, shape=(1,1)):
        """Introduce new variables to the model class. Define variable type, name and shape (optional).

        **Example:**

        ::

            # States struct (optimization variables):
            C_a = model.set_variable(var_type='_x', var_name='C_a', shape=(1,1))
            T_K = model.set_variable(var_type='_x', var_name='T_K', shape=(1,1))

            # Input struct (optimization variables):
            Q_dot = model.set_variable(var_type='_u', var_name='Q_dot')

            # Fixed parameters:
            alpha = model.set_variable(var_type='_p', var_name='alpha')

        .. note:: ``var_type`` allows a shorthand notation e.g. ``_x`` which is equivalent to ``states``.

        :param var_type: Declare the type of the variable. The following types are valid (long or short name is possible):

            ===========================  ===========  ============================
            Long name                    short name   Remark
            ===========================  ===========  ============================
            ``states``                   ``_x``       Required
            ``inputs``                   ``_u``       Required
            ``algebraic``                ``_z``       Optional
            ``parameter``                ``_p``       Optional
            ``timevarying_parameter``    ``_tvp``     Optional
            ===========================  ===========  ============================
        :type var_type: string
        :param var_name: Set a user-defined name for the parameter. The names are reused throughout do_mpc.
        :type var_type: string
        :param shape: Shape of the current variable (optional), defaults to ``1``.
        :type shape: int or tuple of length 2.

        :raises assertion: var_type must be string
        :raises assertion: var_name must be string
        :raises assertion: shape must be tuple or int
        :raises assertion: Cannot call after :py:func:`Model.setup_model`.

        :return: Returns the newly created symbolic variable.
        :rtype: casadi.SX
        """
        assert self.flags['setup'] == False, 'Cannot call .set_variable after .setup_model.'
        assert isinstance(var_type, str), 'var_type must be str, you have: {}'.format(type(var_type))
        assert isinstance(var_name, str), 'var_name must be str, you have: {}'.format(type(var_name))
        assert isinstance(shape, (tuple,int)), 'shape must be tuple or int, you have: {}'.format(type(shape))

        # Get short names:
        var_type =var_type.replace('states', '_x'
            ).replace('inputs', '_u'
            ).replace('algebraic', '_z'
            ).replace('parameter', '_p'
            ).replace('timevarying_parameter', '_tvp')

        # Check validity of var_type:
        assert var_type in ['_x','_u','_z','_p','_tvp'], 'Trying to set non-existing variable var_type: {} with var_name {}'.format(var_type, var_name)
        # Check validity of var_name:
        assert var_name not in [entry_i.name for entry_i in getattr(self,var_type)], 'The variable name {} for type {} already exists.'.format(var_name, var_type)
        # Create variable:
        var = SX.sym(var_name, shape)

        # Extend var list with new entry:
        getattr(self, var_type).append(entry(var_name, sym=var))

        return var

    def set_expression(self, expr_name, expr):
        """Introduce new expression to the model class. Expressions are not required but can be used
        to extract further information from the model. They can also be used for the objective function or constraints.
        Expressions must be formulated with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.

        **Example:**
        Maybe you are interested in monitoring the product of two states?

        ::

            Introduce two scalar states:
            x_1 = model.set_variable('_x', 'x_1')
            x_2 = model.set_variable('_x', 'x_2')

            # Introduce expression:
            model.set_expression('x1x2', x_1*x_2)

        :param expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX

        :raises assertion: expr_name must be str
        :raises assertion: expr must be a casadi SX or MX type
        :raises assertion: Cannot call after :py:func:`Model.setup_model`.

        :return: Returns the newly created expression. Expression can be used e.g. for the RHS.
        :rtype: casadi.SX
        """
        assert self.flags['setup'] == False, 'Cannot call .set_expression after :py:func:`Model.setup_model`'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))

        self._aux_expression.append(entry(expr_name, expr = expr))
        self._aux.append(entry(expr_name, shape=expr.shape))

        return expr

    def set_meas(self, meas_name, expr):
        """Introduce new measurable output to the model class. By default, the model assumes state-feedback (all states are measured outputs).
        Expressions must be formulated with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.

        .. note::

            For moving horizon estimation it is suggested to declare all inputs (``_u``) and a subset of states (``_x``) as
            measurable output. Some other MHE formulations treat inputs separately.

        **Example:**

        ::

            # Introduce states:
            x_meas = model.set_variable('_x', 'x', 3) # 3 measured states (vector)
            x_est = model.set_variable('_x', 'x', 3) # 3 estimated states (vector)
            # and inputs:
            u = model.set_variable('_u', 'u', 2) # 2 inputs (vector)

            # define measurements:
            model.set_meas('x_meas', x_meas)
            model.set_meas('u', u)

        :param expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX

        :raises assertion: expr_name must be str
        :raises assertion: expr must be a casadi SX or MX type
        :raises assertion: Cannot call after .setup_model.

        :return: Returns the newly created measurement expression.
        :rtype: casadi.SX
        """
        assert self.flags['setup'] == False, 'Cannot call .set_meas after :py:func:`Model.setup_model`'
        assert isinstance(meas_name, str), 'meas_name must be str, you have: {}'.format(type(meas_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))

        self._y_expression.append(entry(meas_name, expr = expr))
        self._y.append(entry(meas_name, shape=expr.shape))

        return expr

    def set_rhs(self, var_name, expr):
        """Formulate the right hand side (rhs) of the ODE:

        .. math::

            \\dot{x} = f(\\dots),

        or the update equation in case of discrete dynamics:

        .. math::

            x_{k+1} = f(\\dots).

        Each defined state variable must have a respective equation (of matching dimension)
        for the rhs. Match the rhs with the state by choosing the corresponding names.
        rhs must be formulated with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.

        **Example**:

        ::

            tank_level = model.set_variable('states', 'tank_level')
            tank_temp = model.set_variable('states', 'tank_temp')

            tank_level_next = 0.5*tank_level
            tank_temp_next = ...

            model.set_rhs('tank_level', tank_level_next)
            model.set_rhs('tank_temp', tank_temp_next)

        :param var_name: Name of any of the previously defined states with: ``model.set_variable('states', [NAME])``
        :type var_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX

        :raises assertion: var_name must be str
        :raises assertion: expr must be a casadi SX or MX type
        :raises assertion: var_name must refer to the previously defined states
        :raises assertion: Cannot call after .setup_model.

        :return: None
        :rtype: None
        """
        assert self.flags['setup'] == False, 'Cannot call .set_rhs after .setup_model.'
        assert isinstance(var_name, str), 'var_name must be str, you have: {}'.format(type(var_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))
        _x_names = self.x.keys()
        assert var_name in _x_names, 'var_name must refer to the previously defined states ({}). You have: {}'.format(_x_names, var_name)
        self.rhs_list.extend([{'var_name': var_name, 'expr': expr}])

    def get_variables(self):
        """Method to retrieve the variables of the model.

        .. warning::

            The method is depreciated and will be removed in a later version.
            Please query class attributes (variables) directly.

        This method is convenient when creating the model in a different file
        than the, e.g. the :py:class:`do_mpc.optimizer`. Returns the variables as a list with the following order:

        * ``_x`` (states)

        * ``_u`` (inputs)

        * ``_z`` (algebraic variables)

        * ``_tvp`` (time varying parameters)

        * ``_p`` (uncertain parameters)

        * ``_aux`` (auxiliary expressions)

        * ``_y`` (measurements, measured)

        * ``_y_expression`` (measurements, calculated)

        The method cannot be called prior to :py:func:`model.setup`.

        Variables can also be retrieved independently of this method with, e.g.:

        ::

            x = model.x

        Note that structures can be indexed with the previously defined variable names:

        ::

            _x['var_name']

        :raises assertion: Model was not setup. Finish model creation by calling model.setup_model().

        :return: List of model variables (``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``, ``_aux``, ``_y``, ``_y_expression``)
        :rtype: list
        """
        warnings.warn('This method is depreciated. Please use class getitem method instead. This will become an error in a future release', DeprecationWarning)
        assert self.flags['setup'] == True, 'Model was not setup. Finish model creation by calling model.setup_model().'

        return self._x, self._u, self._z, self._tvp, self._p, self._aux_expression, self._y, self._y_expression


    def setup_model(self):
        """Legacy method.

        .. warning::

            The method is depreciated and will be removed in a later version.
            Please call :py:func:`setup` instead.
        """

        warnings.warn('This method is depreciated. Please use Model.setup instead. This will become an error in a future release', DeprecationWarning)
        self.setup()

    def setup(self):
        """Setup method must be called to finalize the modelling process.
        All required model variables ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p`` must be declared.
        The right hand side expression for ``_x`` must have been set with :py:func:`set_rhs`.

        Sets default measurement function (state feedback) if :py:func:`set_meas` was not called.

        .. warning::

            After calling :py:func:`setup_model`, the model is locked and no further variables,
            expressions etc. can be set.

        :raises assertion: Definition of right hand side (rhs) is incomplete

        :return: None
        :rtype: None
        """

        # Write self._x, self._u, self._z, self._y, self._tvp, self.p with the respective struct_symSX structures.
        self._x = _x = struct_symSX(self._x)
        self._u = _u =  struct_symSX(self._u)
        self._z = _z = struct_symSX(self._z)
        self._p = _p = struct_symSX(self._p)
        self._tvp = _tvp =  struct_symSX(self._tvp)

        # Write self._aux_expression.
        self._aux_expression = struct_SX(self._aux_expression)
        self._aux = struct_symSX(self._aux)

        # Write self._y_expression (measurement equations) as struct_SX symbolic expression structures.
        # Check if it is an empty list (no user input)
        if not self._y_expression:
            self._y_expression = self._x
        else:
            self._y_expression = struct_SX(self._y_expression)
        self._y = struct_symSX(self._y)

        # Create mutable struct_SX with identical structure as self._x to hold the right hand side.
        self._rhs = struct_SX(self._x)

        # Set the expressions in self._rhs with the previously defined SX.sym variables.
        # Check if an expression is set for every state of the system.
        _x_names = set(self._x.keys())
        for rhs_i in self.rhs_list:
            self._rhs[rhs_i['var_name']] = rhs_i['expr']
            _x_names -= set([rhs_i['var_name']])
        assert len(_x_names) == 0, 'Definition of right hand side (rhs) is incomplete. Missing: {}. Use: set_rhs to define expressions.'.format(_x_names)

        # Declare functions for the right hand side and the aux_expressions.
        self._rhs_fun = Function('rhs_fun', [_x, _u, _z, _tvp, _p], [self._rhs])
        self._aux_expression_fun = Function('aux_expression_fun', [_x, _u, _z, _tvp, _p], [self._aux_expression])
        self._meas_fun = Function('meas_fun', [_x, _u, _z, _tvp, _p], [self._y_expression])

        # Create and store some information about the model regarding number of variables for
        # _x, _y, _u, _z, _tvp, _p, _aux
        self.n_x = self._x.shape[0]
        self.n_y = self._y.shape[0]
        self.n_u = self._u.shape[0]
        self.n_z = self._z.shape[0]
        self.n_tvp = self._tvp.shape[0]
        self.n_p = self._p.shape[0]
        self.n_aux = self._aux_expression.shape[0]

        # Remove temporary storage for the symbolic variables. This allows to pickle the class.
        #delattr(self, 'var_list')
        #delattr(self, 'meas_list')
        #delattr(self, 'expr_list')
        delattr(self, 'rhs_list')

        self.flags['setup'] = True
