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
#import casadi as cas
import casadi.tools as castools
import pdb
from do_mpc.tools._casstructure import _SymVar
from typing import Union,Tuple

class Model:
    """The **do-mpc** model class. This class holds the full model description and is at the core of
    :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC` and :py:class:`do_mpc.estimator.Estimator`.
    The :py:class:`Model` class is created with setting the ``model_type`` (continuous or discrete).
    A ``continous`` model consists of an underlying ordinary differential equation (ODE) or differential algebraic equation (DAE):

    .. math::

       \\dot{x}(t) &= f(x(t),u(t),z(t),p(t),p_{\\text{tv}}(t)) + w(t),\\\\
       0 &= g(x(t),u(t),z(t),p(t),p_{\\text{tv}}(t))\\\\
       y &= h(x(t),u(t),z(t),p(t),p_{\\text{tv}}(t)) + v(t)

    whereas a ``discrete`` model consists of a difference equation:

    .. math::

       x_{k+1} &= f(x_k,u_k,z_k,p_k,p_{\\text{tv},k}) + w_k,\\\\
       0 &= g(x_k,u_k,z_k,p_k,p_{\\text{tv},k})\\\\
       y_k &= h(x_k,u_k,z_k,p_k,p_{\\text{tv},k}) + v_k

    The **do-mpc** model can be initiated with either ``SX`` or ``MX`` variable type.
    We refer to the CasADi documentation on the difference of these two types.

    Note:
        ``SX`` vs. ``MX`` in a nutshell: In general use ``SX`` variables (default).
        If your model consists of scalar operations ``SX`` variables will be beneficial.
        Your implementation will most likely only benefit from ``MX`` variables if you use large(r)-scale matrix-vector multiplications.

    Note:
        The option ``symvar_type`` will be inherited to all derived classes (e.g. :py:class:`do_mpc.simulator.Simulator`,
        :py:class:`do_mpc.controller.MPC` and :py:class:`do_mpc.estimator.Estimator`).
        All symbolic variables in these classes will be chosen respectively.


    **Configuration and setup:**

    Configuring and setting up the :py:class:`Model` involves the following steps:

    1. Use :py:func:`set_variable` to introduce new variables to the model.

    2. Optionally introduce "auxiliary" expressions as functions of the previously defined variables with :py:func:`set_expression`. The expressions can be used for monitoring or be reused as constraints, the cost function etc.

    3. Optionally introduce measurement equations with :py:func:`set_meas`. The syntax is identical to :py:func:`set_expression`. By default state-feedback is assumed.

    4. Define the right-hand-side of the `discrete` or `continuous` model as a function of the previously defined variables with :py:func:`set_rhs`. This method must be called once for each introduced state.

    5. Call :py:func:`setup` to finalize the :py:class:`Model`. No further changes are possible afterwards.

    Note:
        All introduced model variables are accessible as **Attributes** of the :py:class:`Model`.
        Use these attributes to query to variables, e.g. to form the cost function in a seperate file for the MPC configuration.

    Args:
        model_type : Set if the model is ``discrete`` or ``continuous``.
        symvar_type : Set if the model is configured with CasADi ``SX`` or ``MX`` variables.

    Raises:
        assertion: model_type must be string
        assertion: model_type must be either discrete or continuous
    """
    def __init__(self, model_type:str=None, symvar_type:str='SX'):
        assert isinstance(model_type, str), 'model_type must be string, you have: {}'.format(type(model_type))
        assert model_type in ['discrete', 'continuous'], 'model_type must be either discrete or continuous, you have: {}'.format(model_type)
        assert symvar_type in ['SX', 'MX'], 'symvar_type must be either SX or MX, you have: {}'.format(symvar_type)


        self.symvar_type = symvar_type
        self.model_type = model_type
        self.sv = _SymVar(symvar_type)

        # Define private class attributes
        self._x =   {'name': [],'var':[]}
        self._u =   {'name': ['default'], 'var': [self.sv.sym('default', (0,0))]}
        self._z =   {'name': ['default'], 'var': [self.sv.sym('default', (0,0))]}
        self._p =   {'name': ['default'], 'var': [self.sv.sym('default', (0,0))]}
        self._tvp = {'name': ['default'], 'var': [self.sv.sym('default', (0,0))]}
        self._aux = {'name': ['default'], 'var': [self.sv.sym('default', (1,1))]}
        # Process noise
        self._w =   {'name': ['default'], 'var': [self.sv.sym('default', (0,0))]}
        # Measurement noise
        self._v =   {'name': ['default'], 'var': [self.sv.sym('default', (0,0))]}
        # Measurements
        self._y =   {'name': ['default'], 'var': [self.sv.sym('default', (0,0))]}

        # Expressions:
        self._aux_expression = [castools.entry('default', expr=castools.DM(0))]
        self._y_expression = []


        self.rhs_list = []
        self.alg_list = [castools.entry('default', expr=[])]

        self.flags = {
            'setup': False,
        }

    def __getstate__(self):
        """
        Returns the state of the :py:class:`Model` for pickling.

        Warnings:
            The :py:class:`Model` class supports pickling only if:

            1. The model is configured with ``SX`` variables.

            2. The model is setup with :py:func:`setup`.
        """
        # Raise exception if model is using MX symvars
        if self.symvar_type == 'MX':
            raise Exception('Pickling of models using MX symvars is not supported.')
        # Raise exception if model is not setup
        if not self.flags['setup']:
            raise Exception('Pickling of unsetup models is not supported.')

        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """
        Sets the state of the :py:class:`Model` for unpickling. Please see :py:func:`__getstate__` for details and restrictions on pickling.
        """
        self.__dict__.update(state)

        # Update expressions with new symbolic variables created when unpickling:
        self._rhs = self._rhs(self._rhs_fun(self._x, self._u, self._z, self._tvp, self._p, self._w))
        self._alg = self._alg(self._alg_fun(self._x, self._u, self._z, self._tvp, self._p, self._w))
        self._aux_expression = self._aux_expression(self._aux_expression_fun(self._x, self._u, self._z, self._tvp, self._p))
        self._y_expression = self._y_expression(self._meas_fun(self._x, self._u, self._z, self._tvp, self._p, self._v))


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
        var_names = ['x','u','z','p','tvp','y','aux', 'w']
        if isinstance(ind, tuple):
            val = []
            for ind_i in ind:
                assert ind_i in var_names, 'The queried variable {} is not valid. Choose from {}.'.format(ind_i, var_names)
                val.append(getattr(self, ind_i))
            return val
        else:
            val = getattr(self,ind)

        return val

    def _getvar(self, var_name):
        """ Function is called from within all property (x, u, z, p, tvp, y, aux, w) getters.
        Not part of the public API.
        """
        if self.flags['setup']:
            return getattr(self, var_name)
        else:
            # Before calling setup the attributes _x,_u,_z etc. are dicts with the keys: name and var.
            sym_dict = getattr(self, var_name)
            if var_name == '_aux_expression':  # or possibly '_aux, depending on what is called in the getter
                # create the required dict from what is currently a 
                sym_dict = {'name':[entry.name for entry in sym_dict], 
                            'var': [entry.expr for entry in sym_dict]}
            # We use the same method as in setup to create symbolic structures from these dicts
            sym_struct = self._convert2struct(sym_dict)
            # We then create a mutable structure of the same structure
            struct = self.sv.struct(sym_struct)
            # And set the values of this structure to the original symbolic variables
            for key, var in zip(sym_dict['name'], sym_dict['var']):
                struct[key] = var
            # Indexing this structure returns the original symbolic variables
            return struct


    @property
    def x(self):
        """ Dynamic states.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            Note:
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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            Raises:
                assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_x')

    @x.setter
    def x(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

    @property
    def u(self):
        """ Inputs.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            Note:
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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            Raises:
                assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_u')

    @u.setter
    def u(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

    @property
    def z(self):
        """ Algebraic states.
        CasADi symbolic structure, can be indexed with user-defined variable names.

        Note:
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

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        Raises:
            assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_z')

    @z.setter
    def z(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

    @property
    def p(self):
        """ Static parameters.
        CasADi symbolic structure, can be indexed with user-defined variable names.

        Note:
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

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        Raises:
            assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_p')

    @p.setter
    def p(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

    @property
    def tvp(self):
        """ Time-varying parameters.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            Note:
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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            Raises:
                assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_tvp')

    @tvp.setter
    def tvp(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

    @property
    def y(self):
        """ Measurements.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            Note:
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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            Raises:
                assertion: Cannot set model variables directly Use set_meas instead.
        """
        return self._getvar('_y')

    @y.setter
    def y(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

    @property
    def aux(self):
        """ Auxiliary expressions.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            Note:
                Expressions are introduced with :py:func:`Model.set_expression` Use this property only to query
                variables.

            **Example:**

            ::

                model = do_mpc.model.Model('continuous')
                model.set_variable('_x','temperature', 4) # 4 states
                dt = model.x['temperature',0]- model.x['temperature', 1]
                model.set_expression('dtemp', dt)
                # Query:
                model.aux['dtemp', 0] # 0th element of variable
                model.aux['dtemp']    # all elements of variable

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            Raises:
                assertion: Cannot set aux directly Use set_expression instead.
        """
        return self._getvar('_aux_expression')

    @aux.setter
    def aux(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

    @property
    def w(self):
        """ Process noise.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            The process noise structure is created automatically, whenever the
            :py:func:`Model.set_rhs` method is called with the argument ``process_noise = True``.

            Note:
                The process noise is used for the :py:class:`do_mpc.estimator.MHE` and
                can be used to simulate a disturbed system in the :py:class:`do_mpc.simulator.Simulator`.

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            Raises:
                assertion: Cannot set w directly
        """
        return self._getvar('_w')

    @w.setter
    def w(self, val):
        raise Exception('Cannot set process noise directly.')


    @property
    def v(self):
        """ Measurement noise.
            CasADi symbolic structure, can be indexed with user-defined variable names.

            The measurement noise structure is created automatically, whenever the
            :py:func:`Model.set_meas` method is called with the argument ``meas_noise = True``.

            Note:
                The measurement noise is used for the :py:class:`do_mpc.estimator.MHE` and
                can be used to simulate a disturbed system in the :py:class:`do_mpc.simulator.Simulator`.

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            Raises:
                assertion: Cannot set v directly
        """
        return self._getvar('_v')

    @v.setter
    def v(self, val):
        raise Exception('Cannot set measurement noise directly.')


    def set_variable(self, var_type:str, var_name:str, shape:Union[int,Tuple]=(1,1))->Union[castools.SX,castools.MX]:
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

        Note: 
            ``var_type`` allows a shorthand notation e.g. ``_x`` which is equivalent to ``states``.

        Args:
            var_type : Declare the type of the variable.
            var_name : Set a user-defined name for the parameter. The names are reused throughout do_mpc.
            shape : Shape of the current variable (optional), defaults to ``1``. 
               
        The following types of **var_type** are valid (long or short name is possible):

        ===========================  ===========  ============================
        Long name                    short name   Remark
        ===========================  ===========  ============================
        ``states``                   ``_x``       Required
        ``inputs``                   ``_u``       optional
        ``algebraic``                ``_z``       Optional
        ``parameter``                ``_p``       Optional
        ``timevarying_parameter``    ``_tvp``     Optional
        ===========================  ===========  ============================

        Raises:
            assertion: **var_type** must be string
            assertion: **var_name** must be string
            assertion: **shape** must be tuple or int
            assertion: Cannot call after :py:func:`setup`.

        Returns:
            Returns the newly created symbolic variable.
        """
        assert self.flags['setup'] == False, 'Cannot call .set_variable after setup.'
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
        assert var_name not in getattr(self,var_type)['name'], 'The variable name {} for type {} already exists.'.format(var_name, var_type)

        # Create variable:
        var = self.sv.sym(var_name, shape)


        # Extend var list with new entry:
        getattr(self, var_type)['var'].append(var)
        getattr(self, var_type)['name'].append(var_name)

        return var

    def set_expression(self, expr_name:str, expr:Union[castools.SX,castools.MX])->Union[castools.SX,castools.MX]:
        """Introduce new expression to the model class. Expressions are not required but can be used
        to extract further information from the model.
        Expressions must be formulated with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.

        **Example:**

        Maybe you are interested in monitoring the product of two states?

        ::

            Introduce two scalar states:
            x_1 = model.set_variable('_x', 'x_1')
            x_2 = model.set_variable('_x', 'x_2')

            # Introduce expression:
            model.set_expression('x1x2', x_1*x_2)

        This new expression ``x1x2`` is then available in all **do-mpc** modules utilizing
        this model instance. It can be set, e.g. as the cost function in :py:class:`do_mpc.controller.MPC`
        or simply used in a graphical representation of the simulated / controlled system.

        Args:
            expr_name : Arbitrary name for the given expression. Names are used for key word indexing.
            expr : CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.

        Raises:
            assertion: expr_name must be str
            assertion: expr must be a casadi SX or MX type
            assertion: Cannot call after :py:func:`setup`.

        Returns:
            Returns the newly created expression. Expression can be used e.g. for the RHS.
        """
        assert self.flags['setup'] == False, 'Cannot call .set_expression after setup'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (castools.SX, castools.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))

        self._aux_expression.append(castools.entry(expr_name, expr = expr))

        # Create variable:
        var = self.sv.sym(expr_name, expr.shape)
        self._aux['var'].append(var)
        self._aux['name'].append(expr_name)

        return expr

    def set_meas(self, meas_name:str, expr:Union[castools.SX,castools.MX], meas_noise:bool=True)->Union[castools.SX,castools.MX]:
        """Introduce new measurable output to the model class.

        .. math::

            y = h(x(t),u(t),z(t),p(t),p_{\\text{tv}}(t)) + v(t)

        or in case of discrete dynamics:

        .. math::

            y_k = h(x_k,u_k,z_k,p_k,p_{\\text{tv},k}) + v_k

        By default, the model assumes state-feedback (all states are measured outputs).
        Expressions must be formulated with respect to ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.

        Be default, it is assumed that the measurements experience additive noise :math:`v_k`.
        This can be deactivated for individual measured variables by changing the boolean variable
        ``meas_noise`` to ``False``.
        Note that measurement noise is only meaningful for state-estimation and will not affect the controller.
        Furthermore, it can be set with each :py:class:`do_mpc.simulator.Simulator` call to obtain imperfect outputs.

        Note:
            For moving horizon estimation it is suggested to declare all inputs (``_u``) and e.g. a subset of states (``_x``) as
            measurable output. Some other MHE formulations treat inputs separately.

        Note:
            It is often suggested to deactivate measurement noise for "measured" inputs (``_u``).
            These can typically seen as certain variables.

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

        Args:
            meas_name : Arbitrary name for the given expression. Names are used for key word indexing.
            expr : CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
            meas_noise : Set if the measurement equation is disturbed by additive noise.

        Raises:
            assertion: expr_name must be str
            assertion: expr must be a casadi SX or MX type
            assertion: Cannot call after :py:func:`setup`.

        Returns:
            Returns the newly created measurement expression.
        """
        assert self.flags['setup'] == False, 'Cannot call .set_meas after setup'
        assert isinstance(meas_name, str), 'meas_name must be str, you have: {}'.format(type(meas_name))
        assert isinstance(expr, (castools.SX, castools.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))
        assert isinstance(meas_noise, bool), 'meas_noise must be of type boolean. You have: {}'.format(type(meas_noise))

        # Create a new process noise variable and add it to the rhs equation.
        if meas_noise:
            var = self.sv.sym(meas_name+'_noise', expr.shape[0])

            self._v['name'].append(meas_name+'_noise')
            self._v['var'].append(var)
            expr += var

        self._y_expression.append(castools.entry(meas_name, expr = expr))

        # Create variable:
        var = self.sv.sym(meas_name, expr.shape)
        self._y['var'].append(var)
        self._y['name'].append(meas_name)

        return expr

    def set_rhs(self, var_name:str, expr:Union[castools.SX,castools.MX], process_noise:bool=False)->None:
        """Formulate the right hand side (rhs) of the ODE:

        .. math::

            \\dot{x}(t) = f(x(t),u(t),z(t),p(t),p_{\\text{tv}}(t)) + w(t),

        or the update equation in case of discrete dynamics:

        .. math::

            x_{k+1} = f(x_k,u_k,z_k,p_k,p_{\\text{tv},k}) + w_k,

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

        Optionally, set ``process_noise = True`` to introduce an additive process noise variable.
        This is  meaningful for the :py:class:`do_mpc.estimator.MHE` (See :py:func:`do_mpc.estimator.MHE.set_default_objective` for more details).
        Furthermore, it can be set with each :py:class:`do_mpc.simulator.Simulator` call to obtain imperfect (realistic) simulation results.

        Args:
            var_name : Reference to previously introduced state names (with :py:func:`Model.set_variable`)
            expr : CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
            process_noise : Make the respective state variable non-deterministic.

        Raises:
            assertion: var_name must be str
            assertion: expr must be a casadi SX or MX type
            assertion: var_name must refer to the previously defined states
            assertion: Cannot call after :py:func`setup`.
        """
        assert self.flags['setup'] == False, 'Cannot call .set_rhs after .setup.'
        assert isinstance(var_name, str), 'var_name must be str, you have: {}'.format(type(var_name))
        assert isinstance(expr, (castools.SX, castools.MX, castools.DM)), 'expr must be a casadi SX, MX or DM type, you have:{}'.format(type(expr))
        assert var_name in self._x['name'], 'var_name must refer to the previously defined states ({}). You have: {}'.format(self._x['name'], var_name)

        # Create a new process noise variable and add it to the rhs equation.
        if process_noise:
            if self.symvar_type == 'MX':
                var = castools.MX.sym(var_name+'_noise', expr.shape[0])
            else:
                var = castools.SX.sym(var_name+'_noise', expr.shape[0])

            self._w['name'].append(var_name + '_noise')
            self._w['var'].append(var)
            expr += var
        self.rhs_list.extend([{'var_name': var_name, 'expr': expr}])

    def set_alg(self, expr_name:str, expr:Union[castools.SX,castools.MX])->None:
        """ Introduce new algebraic equation to model.

        For the continous time model, the expression must be formulated as

        .. math::

           0 = g(x(t),u(t),z(t),p(t),p_{\\text{tv}}(t))


        or for a ``discrete`` model:

        .. math::

           0 = g(x_k,u_k,z_k,p_k,p_{\\text{tv},k})

        Note:
            For the introduced algebraic variables :math:`z \in \mathbb{R}^{n_z}`
            it is required to introduce exactly :math:`n_z` algebraic equations.
            Otherwise :py:meth:`setup` will throw an error message.

        Args:
            expr_name : Name of the introduced expression
            expr : CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        """
        assert self.flags['setup'] == False, 'Cannot call .set_alg after .setup.'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (castools.SX, castools.MX, castools.DM)), 'expr must be a casadi SX, MX or DM type, you have:{}'.format(type(expr))

        self.alg_list.append(castools.entry(expr_name, expr = expr))


    def _convert2struct(self, var_dict:dict)->Union[castools.structure3.SXStruct,castools.structure3.MXStruct]:
        """Helper function for :py:func:`setup`. Not part of the public API.
        This method is used to convert the attributes:

        ::
            self._x
            self._u
            ...

        into structures of type ``struct_symSX`` or ``struct_symMX`` (depending on the attribute ``symvar_type``).

        These structures are created with **newly introduced** symbolic variables which are of the same shapes and names
        as those introduced with :py:func:`set_variable`.

        **Why is this necessary?**

        For the symbolic variable type ``MX`` it is impossible to first create symbolic variables and then combine them into a structure (``struct_symMX``).
        We thus create symbolic variables, then create a structure holding similar variables and the substitute these newly introduced variables in all expressions.

        Args:
            var_dict : Attributes that are configured with :py:func:`set_variable` (e.g. ``self._x``). These attributes are of type ``dict`` with the keys ``name`` and ``var``
        
        Returns:
            CasADi structure
        """
        result_struct =  self.sv.sym_struct([
            castools.entry(name, shape = var.shape) for var, name in zip(var_dict['var'], var_dict['name'])
        ])
        return result_struct

    def _substitute_struct_vars(self, var_dict_list:list, sym_struct_list:list, expr:Union[castools.structure3.SXStruct,castools.structure3.MXStruct]):
        """Helper function for :py:func:`setup`. Not part of the public API.
        This method is used to substitute the newly introduced structured variables with :py:func:`_convert2struct`
        into the expressions that define the model (e.g. ``_rhs``).

        **Why is this necessary?**

        For the symbolic variable type ``MX`` it is impossible to first create symbolic variables and then combine them into a structure (``struct_symMX``).
        We thus create symbolic variables, then create a structure holding similar variables and the substitute these newly introduced variables in all expressions.

        Args:
            var_dict_list : List of attributes that are configured with :py:func:`set_variable` (e.g. ``self._x``). These attributes are of type ``dict`` with the keys ``name`` and ``var``
            sym_dict_list : List of the same attributes converted into structures with :py:func:`_convert2struct`.
            expr: Casadi structured expr in which the variables from ``var_dict_list`` are substituted with those from ``sym_struct_list``.
        """
        assert len(var_dict_list)==len(sym_struct_list)

        subs = expr
        for var_dict, sym_struct in zip(var_dict_list, sym_struct_list):
            assert var_dict['name'] == sym_struct.keys()

            for var, name in zip(var_dict['var'], var_dict['name']):
                subs = castools.substitute(subs, var, sym_struct[name])

        if self.symvar_type == 'MX':
            expr = expr(subs)
        else:
            expr.master = subs
        
        return expr

    def _substitute_exported_vars(self, var_dict_list, sym_struct_list):
        """Helper function for :py:func:`setup`. Not part of the public API.
        This method is used to substitute the newly introduced structured variables with :py:func:`_convert2struct`
        in all previously EXPORTED variables, e.g. with :py:func:`set_variable`.

        This is necessary because otherwise variables obtained from the model PRIOR to calling :py:func:`setup`
        are not the same as those returned after calling :py:func:`setup`:

        ::

            x = model.set_variable('_x', 'x')

            ...

            model.setup()

            # We don't want this:
            model.x['x'] == x
            >> MX(x==x)

            # We want this:
            model.x['x'] == x
            >> MX(1)

        This is a bit of a HACKY solution and might require fixing if CasADi is changing its API.
        """
        for var_dict, sym_struct in zip(var_dict_list, sym_struct_list):
            assert var_dict['name'] == sym_struct.keys()

            for var, name in zip(var_dict['var'], var_dict['name']):
                var.__dict__['this'] = sym_struct[name].__dict__['this']


    def setup(self)->None:
        """Setup method must be called to finalize the modelling process.
        All required model variables must be declared.
        The right hand side expression for ``_x`` must have been set with :py:func:`set_rhs`.

        Sets default measurement function (state feedback) if :py:func:`set_meas` was not called.

        Warnings:
            After calling :py:func:`setup`, the model is locked and no further variables,
            expressions etc. can be set.

        Raises:
            assertion: Definition of right hand side (rhs) is incomplete
        """
        # Set all states as measurements if set_meas was not called by user.
        if not self._y_expression:
            for name, var in zip(self._x['name'], self._x['var']):
                self.set_meas(name, var)

        # Write self._y_expression (measurement equations) as struct symbolic expression structures.
        self._y_expression = self.sv.struct(self._y_expression)

        # Create structure from listed symbolic variables:
        _x =  self._convert2struct(self._x)
        _w =  self._convert2struct(self._w)
        _v =  self._convert2struct(self._v)
        _u =  self._convert2struct(self._u)
        _z =  self._convert2struct(self._z)
        _p =  self._convert2struct(self._p)
        _tvp =  self._convert2struct(self._tvp)
        _aux =  self._convert2struct(self._aux)
        _y =  self._convert2struct(self._y)

        # Write self._aux_expression.
        self._aux_expression = self.sv.struct(self._aux_expression)


        # Create alg equations:
        self._alg = self.sv.struct(self.alg_list)

        # Create mutable struct with identical structure as _x to hold the right hand side.
        self._rhs = self.sv.struct(_x)

        # Set the expressions in self._rhs with the previously defined SX.sym variables.
        # Check if an expression is set for every state of the system.
        _x_names = set(self._x['name'])
        for rhs_i in self.rhs_list:
            self._rhs[rhs_i['var_name']] = rhs_i['expr']
            _x_names -= set([rhs_i['var_name']])
        assert len(_x_names) == 0, 'Definition of right hand side (rhs) is incomplete. Missing: {}. Use: set_rhs to define expressions.'.format(_x_names)

        var_dict_list = [self._x, self._w, self._v, self._u, self._z, self._p, self._tvp]
        sym_struct_list = [_x, _w, _v, _u, _z, _p, _tvp]

        self._rhs = self._substitute_struct_vars(var_dict_list, sym_struct_list, self._rhs)
        self._alg = self._substitute_struct_vars(var_dict_list, sym_struct_list, self._alg)
        self._aux_expression = self._substitute_struct_vars(var_dict_list, sym_struct_list, self._aux_expression)
        self._y_expression = self._substitute_struct_vars(var_dict_list, sym_struct_list, self._y_expression)

        self._substitute_exported_vars(var_dict_list, sym_struct_list)

        self._x = _x
        self._w = _w
        self._v = _v
        self._u = _u
        self._z = _z
        self._p = _p
        self._tvp = _tvp
        self._y = _y
        self._aux = _aux
        
        A_lin_expr = castools.jacobian(self._rhs,self._x)
        B_lin_expr = castools.jacobian(self._rhs,self._u)
        C_lin_expr = castools.jacobian(self._y_expression,self._x)
        D_lin_expr = castools.jacobian(self._y_expression,self._u)

        # Declare functions for the right hand side and the aux_expressions.
        self._rhs_fun = castools.Function('rhs_fun',
                                 [_x, _u, _z, _tvp, _p, _w], [self._rhs],
                                 ["_x", "_u", "_z", "_tvp", "_p", "_w"], ["_rhs"])
        self._alg_fun = castools.Function('alg_fun',
                                 [_x, _u, _z, _tvp, _p, _w], [self._alg],
                                 ["_x", "_u", "_z", "_tvp", "_p", "_w"], ["_alg"])
        self._aux_expression_fun = castools.Function('aux_expression_fun',
                                            [_x, _u, _z, _tvp, _p], [self._aux_expression],
                                            ["_x", "_u", "_z", "_tvp", "_p"], ["_aux_expression"])
        self._meas_fun = castools.Function('meas_fun',
                                  [_x, _u, _z, _tvp, _p, _v], [self._y_expression],
                                  ["_x", "_u", "_z", "_tvp", "_p", "_v"], ["_y_expression"])
        self.A_fun = castools.Function('A_fun',
                              [_x, _u, _z, _tvp, _p, _w],[A_lin_expr],
                              ["_x", "_u", "_z", "_tvp", "_p", "_w"],["A_lin_expr"])
        self.B_fun = castools.Function('B_fun',
                              [_x, _u, _z, _tvp, _p, _w],[B_lin_expr],
                              ["_x", "_u", "_z", "_tvp", "_p", "_w"],["B_lin_expr"])
        self.C_fun = castools.Function('C_fun',
                              [_x, _u, _z, _tvp, _p, _v],[C_lin_expr],
                              ["_x", "_u", "_z", "_tvp", "_p", "_v"],["C_lin_expr"])
        self.D_fun = castools.Function('D_fun',
                              [_x, _u, _z, _tvp, _p, _v],[D_lin_expr],
                              ["_x", "_u", "_z", "_tvp", "_p", "_v"],["D_lin_expr"]) 

        # Create and store some information about the model regarding number of variables for
        # _x, _y, _u, _z, _tvp, _p, _aux
        self.n_x = self._x.shape[0]
        self.n_y = self._y.shape[0]
        self.n_u = self._u.shape[0]
        self.n_z = self._z.shape[0]
        self.n_tvp = self._tvp.shape[0]
        self.n_p = self._p.shape[0]
        self.n_aux = self._aux_expression.shape[0]
        self.n_w = self._w.shape[0]
        self.n_v = self._v.shape[0]

        msg = 'Must have the same number of algebraic equations (you have {}) and variables (you have {}).'
        assert self.n_z == self._alg.shape[0], msg.format(self._alg.shape[0], self.n_z)

        # Remove temporary storage for the symbolic variables. This allows to pickle the class.
        delattr(self, 'rhs_list')
        delattr(self, 'alg_list')

        self.flags['setup'] = True


    @staticmethod
    def _transfer_variables(old_model, new_model, transfer=['_x', '_u', '_tvp', '_p']):
        """Private and static method to transfer variables from old model to new model.
        This is used in :func:`do_mpc.model.linearize` and :meth:`LinearModel.discretize`.

        Extracts information about the noise (measurement and process) from the old model and
        returns an array of booleans for the measurements and states (one element for each named instance).
        """
        # Initialize array for process noise and measurement noise
        process_noise = old_model.x(False)
        meas_noise = old_model.y(False)

        
        #Setting states and inputs (get rid of defaults for u, tvp, p)
        for key in old_model.x.keys():
            new_model.set_variable('_x', key, shape=old_model.x[key].shape)
        for key in old_model.u.keys()[1:]:
            new_model.set_variable('_u', key, shape=old_model.u[key].shape)
        for key in old_model.tvp.keys()[1:]:
            new_model.set_variable('_tvp', key, shape=old_model.tvp[key].shape)
        for key in old_model.p.keys()[1:]:
            new_model.set_variable('_p', key, shape=old_model.p[key].shape)
        for key in old_model.aux.keys()[1:]:
            expr_fun = old_model._aux_expression_fun
            expr = expr_fun(new_model.x, new_model.u, new_model.z, new_model.tvp, new_model.p)
            expr_struct = old_model._aux_expression(expr)
            new_model.set_expression(key, expr_struct[key])          
        
    
    def get_linear_system_matrices(self, 
                                   xss:np.ndarray=None, 
                                   uss:np.ndarray=None, 
                                   z:np.ndarray=None, 
                                   tvp:np.ndarray=None,
                                   p:np.ndarray=None
                                   )->Union[Tuple[castools.SX,castools.SX,castools.SX,castools.SX],Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]]:
        """
        Returns the matrix quadrupel :math:`(A,B,C,D)` of the linearized system around the operating point (``xss,uss,z,tvp,p,w,v``).
        All arguments are optional in which case the matrices might still be symbolic. 
        If the matrices are not symbolic, they are returned as numpy arrays.
        
        Args:
            xss : Steady state state
            uss : Steady state input
            z : Steady state algebraic states
            tvp : time varying parameters set point
            p : parameters set point
        
        Returns:
            State matrix, Input matrix, Output matrix, Feedforward matrix
        """
        if self.symvar_type == 'MX':
            raise ValueError("get_linear_system_matrices requires symvar_type SX")
        # Default values for all variables are the symbolic variables themselves.
        if isinstance(xss, type(None)):
            xss = self.x
        if isinstance(uss, type(None)):
            uss = self.u
        if isinstance(z, type(None)):
            z = self.z
        if isinstance(tvp, type(None)):
            tvp = self.tvp
        if isinstance(p, type(None)):
            p = self.p
        
        # Noise variables are always set to zero.
        w = self.w(0)
        v = self.v(0)

        # Create A,B,C,D matrices and check if (for the given inputs) they are constant are still symbolic.
        # Constant matrices are converted to numpy arrays.
        A = self.A_fun(xss, uss, z, tvp, p, w)
        if A.is_constant():
            A = castools.DM(A).full()
        B = self.B_fun(xss, uss, z, tvp, p, w)
        if B.is_constant():
            B = castools.DM(B).full()
        C = self.C_fun(xss, uss, z, tvp, p, v)
        if C.is_constant():
            C = castools.DM(C).full()
        D = self.D_fun(xss, uss, z, tvp, p, v)
        if D.is_constant():
            D = castools.DM(D).full()
        return A,B,C,D