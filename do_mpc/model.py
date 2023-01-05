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
from do_mpc.tools.casstructure import _SymVar, _struct_MX, _struct_SX
from scipy.signal import cont2discrete


class IteratedVariables:
    """ Class to initiate properties and attributes for iterated variables.
    This class is inherited to all iterating **do-mpc** classes and based on the :py:class:`Model`.

    .. warning::

        This base class can not be used independently.
    """

    def __init__(self):
        assert 'model' in self.__dict__.keys(), 'Cannot initialize variables before assigning the model to the current class instance.'

        # Initialize structure for intial conditions:
        self._x0 = self.model._x(0.0)
        self._u0 = self.model._u(0.0)
        self._z0 = self.model._z(0.0)
        self._t0 = np.array([0.0])


    def _convert2struct(self, val, struct):
        """ Convert array to structure.
        Pass ``val`` which can be an int, float, array, structure and return
        a numerical structure based on the second argument ``structure``.

        If a structure is passed, return the structure unchanged.

        Performs some sanity checks.
        """

        # convert to array
        if isinstance(val, (float, int)):
            val = np.array([val])

        # Check dimensions
        err_msg = 'Variable cannot be set because the supplied vector has the wrong size. You have {} and the model is setup for {}'
        n_val = np.prod(val.shape)
        n_var = struct.size
        assert n_val == n_var, err_msg.format(n_val, n_var)

        # Convert to structure (or return structure)
        if isinstance(val, (np.ndarray, casadi.DM)):
            val = struct(val)
        elif isinstance(val, structure3.DMStruct):
            pass
        else:
            types = (np.ndarray, casadi.DM, structure3.DMStruct)
            raise Exception('x0 must be of tpye {}. You have: {}'.format(types, type(val)))

        return val

    @property
    def x0(self):
        """ Initial state and current iterate.
        This is the numerical structure holding the information about the current states
        in the class. The property can be indexed according to the model definition.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_x','temperature', shape=(4,1))

            ...
            mhe = do_mpc.estimator.MHE(model)
            # or
            mpc = do_mpc.estimator.MPC(model)

            # Get or set current value of variable:
            mpc.x0['temperature', 0] # 0th element of variable
            mpc.x0['temperature']    # all elements of variable
            mpc.x0['temperature', 0:2]    # 0th and 1st element

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        """
        return self._x0

    @x0.setter
    def x0(self, val):
        self._x0 = self._convert2struct(val, self.model._x)

    @property
    def u0(self):
        """ Initial input and current iterate.
        This is the numerical structure holding the information about the current input
        in the class. The property can be indexed according to the model definition.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_u','heating', shape=(4,1))

            ...
            mhe = do_mpc.estimator.MHE(model)
            # or
            mpc = do_mpc.estimator.MPC(model)

            # Get or set current value of variable:
            mpc.u0['heating', 0] # 0th element of variable
            mpc.u0['heating']    # all elements of variable
            mpc.u0['heating', 0:2]    # 0th and 1st element

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        """
        return self._u0

    @u0.setter
    def u0(self, val):
        self._u0 = self._convert2struct(val, self.model._u)

    @property
    def z0(self):
        """ Initial algebraic state and current iterate.
        This is the numerical structure holding the information about the current algebraic states
        in the class. The property can be indexed according to the model definition.

        **Example:**

        ::

            model = do_mpc.model.Model('continuous')
            model.set_variable('_z','temperature', shape=(4,1))

            ...
            mhe = do_mpc.estimator.MHE(model)
            # or
            mpc = do_mpc.estimator.MPC(model)

            # Get or set current value of variable:
            mpc.z0['temperature', 0] # 0th element of variable
            mpc.z0['temperature']    # all elements of variable
            mpc.z0['temperature', 0:2]    # 0th and 1st element

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``

        """
        return self._z0

    @z0.setter
    def z0(self, val):
        self._z0 = self._convert2struct(val, self.model._z)

    @property
    def t0(self):
        """ Current time marker of the class.
        Use this property to set of query the time.

        Set with ``int``, ``float``, ``numpy.ndarray`` or ``casadi.DM`` type.
        """
        return self._t0

    @t0.setter
    def t0(self,val):
        if isinstance(val, (int,float)):
            self._t0 = np.array([val])
        elif isinstance(val, np.ndarray):
            assert val.size == 1, 'Cant set time with shape {}. Must contain exactly one element.'.format(val.size)
            self._t0 = val.flatten()
        elif isinstance(val, casadi.DM):
            assert val.size == 1, 'Cant set time with shape {}. Must contain exactly one element.'.format(val.size)
            self._t0 = val.full().flatten()
        else:
            types = (np.ndarray, float, int, casadi.DM)
            raise Exception('Passing object of type {} to set the current time. Must be of type {}'.format(type(val), types))


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

    .. note::

        ``SX`` vs. ``MX`` in a nutshell: In general use ``SX`` variables (default).
        If your model consists of scalar operations ``SX`` variables will be beneficial.
        Your implementation will most likely only benefit from ``MX`` variables if you use large(r)-scale matrix-vector multiplications.

    .. note::

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

    .. note::

        All introduced model variables are accessible as **Attributes** of the :py:class:`Model`.
        Use these attributes to query to variables, e.g. to form the cost function in a seperate file for the MPC configuration.

    :param model_type: Set if the model is ``discrete`` or ``continuous``.
    :type model_type: str
    :param symvar_type: Set if the model is configured with CasADi ``SX`` or ``MX`` variables.
    :type symvar_type: str

    :raises assertion: model_type must be string
    :raises assertion: model_type must be either discrete or continuous

    .. automethod:: __getitem__
    """

    def __init__(self, model_type=None, symvar_type='SX'):
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
        self._aux_expression = [entry('default', expr=DM(0))]
        self._y_expression = []


        self.rhs_list = []
        self.alg_list = [entry('default', expr=[])]

        self.flags = {
            'setup': False,
        }

    def __getstate__(self):
        """
        Returns the state of the :py:class:`Model` for pickling.

        .. warning::

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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``


            :raises assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_x')

    @x.setter
    def x(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_u')

    @u.setter
    def u(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

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

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``


        :raises assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_z')

    @z.setter
    def z(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

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

        Useful CasADi symbolic structure methods:

        * ``.shape``

        * ``.keys()``

        * ``.labels()``


        :raises assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_p')

    @p.setter
    def p(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set model variables directly Use set_variable instead.
        """
        return self._getvar('_tvp')

    @tvp.setter
    def tvp(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

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

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set model variables directly Use set_meas instead.
        """
        return self._getvar('_y')

    @y.setter
    def y(self, val):
        raise Exception('Cannot set model variables directly Use set_variable instead.')

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
                model.aux['dtemp', 0] # 0th element of variable
                model.aux['dtemp']    # all elements of variable

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set aux directly Use set_expression instead.
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

            .. note::

                The process noise is used for the :py:class:`do_mpc.estimator.MHE` and
                can be used to simulate a disturbed system in the :py:class:`do_mpc.simulator.Simulator`.

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set w directly
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

            .. note::

                The measurement noise is used for the :py:class:`do_mpc.estimator.MHE` and
                can be used to simulate a disturbed system in the :py:class:`do_mpc.simulator.Simulator`.

            Useful CasADi symbolic structure methods:

            * ``.shape``

            * ``.keys()``

            * ``.labels()``

            :raises assertion: Cannot set v directly
        """
        return self._getvar('_v')

    @v.setter
    def v(self, val):
        raise Exception('Cannot set measurement noise directly.')


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
            ``inputs``                   ``_u``       optional
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
        :raises assertion: Cannot call after :py:func:`setup`.

        :return: Returns the newly created symbolic variable.
        :rtype: casadi.SX
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

    def set_expression(self, expr_name, expr):
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
        this model instance. It can be set, e.g. as the cost function in :py:class:`do-mpc.controller.MPC`
        or simply used in a graphical representation of the simulated / controlled system.

        :param expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX

        :raises assertion: expr_name must be str
        :raises assertion: expr must be a casadi SX or MX type
        :raises assertion: Cannot call after :py:func:`setup`.

        :return: Returns the newly created expression. Expression can be used e.g. for the RHS.
        :rtype: casadi.SX
        """
        assert self.flags['setup'] == False, 'Cannot call .set_expression after setup'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))

        self._aux_expression.append(entry(expr_name, expr = expr))

        # Create variable:
        var = self.sv.sym(expr_name, expr.shape)
        self._aux['var'].append(var)
        self._aux['name'].append(expr_name)

        return expr

    def set_meas(self, meas_name, expr, meas_noise=True):
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

        .. note::

            For moving horizon estimation it is suggested to declare all inputs (``_u``) and e.g. a subset of states (``_x``) as
            measurable output. Some other MHE formulations treat inputs separately.

        .. note::

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

        :param expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX
        :param meas_noise: Set if the measurement equation is disturbed by additive noise.
        :type meas_noise: bool

        :raises assertion: expr_name must be str
        :raises assertion: expr must be a casadi SX or MX type
        :raises assertion: Cannot call after :py:func:`setup`.

        :return: Returns the newly created measurement expression.
        :rtype: casadi.SX
        """
        assert self.flags['setup'] == False, 'Cannot call .set_meas after setup'
        assert isinstance(meas_name, str), 'meas_name must be str, you have: {}'.format(type(meas_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))
        assert isinstance(meas_noise, bool), 'meas_noise must be of type boolean. You have: {}'.format(type(meas_noise))

        # Create a new process noise variable and add it to the rhs equation.
        if meas_noise:
            var = self.sv.sym(meas_name+'_noise', expr.shape[0])

            self._v['name'].append(meas_name+'_noise')
            self._v['var'].append(var)
            expr += var

        self._y_expression.append(entry(meas_name, expr = expr))

        # Create variable:
        var = self.sv.sym(meas_name, expr.shape)
        self._y['var'].append(var)
        self._y['name'].append(meas_name)

        return expr

    def set_rhs(self, var_name, expr, process_noise=False):
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


        :param var_name: Reference to previously introduced state names (with :py:func:`Model.set_variable`)
        :type var_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX
        :param process_noise: (optional) Make the respective state variable non-deterministic.
        :type process_noise: boolean

        :raises assertion: var_name must be str
        :raises assertion: expr must be a casadi SX or MX type
        :raises assertion: var_name must refer to the previously defined states
        :raises assertion: Cannot call after :py:func`setup`.

        :return: None
        :rtype: None
        """
        assert self.flags['setup'] == False, 'Cannot call .set_rhs after .setup.'
        assert isinstance(var_name, str), 'var_name must be str, you have: {}'.format(type(var_name))
        assert isinstance(expr, (casadi.SX, casadi.MX, casadi.DM)), 'expr must be a casadi SX, MX or DM type, you have:{}'.format(type(expr))
        assert var_name in self._x['name'], 'var_name must refer to the previously defined states ({}). You have: {}'.format(self._x['name'], var_name)

        # Create a new process noise variable and add it to the rhs equation.
        if process_noise:
            if self.symvar_type == 'MX':
                var = MX.sym(var_name+'_noise', expr.shape[0])
            else:
                var = SX.sym(var_name+'_noise', expr.shape[0])

            self._w['name'].append(var_name + '_noise')
            self._w['var'].append(var)
            expr += var
        self.rhs_list.extend([{'var_name': var_name, 'expr': expr}])

    def set_alg(self, expr_name, expr):
        """ Introduce new algebraic equation to model.

        For the continous time model, the expression must be formulated as

        .. math::

           0 = g(x(t),u(t),z(t),p(t),p_{\\text{tv}}(t))


        or for a ``discrete`` model:

        .. math::

           0 = g(x_k,u_k,z_k,p_k,p_{\\text{tv},k})

        .. note::

            For the introduced algebraic variables :math:`z \in \mathbb{R}^{n_z}`
            it is required to introduce exactly :math:`n_z` algebraic equations.
            Otherwise :py:meth:`setup` will throw an error message.

        :param expr_name: Name of the introduced expression
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on ``_x``, ``_u``, ``_z``, ``_tvp``, ``_p``.
        :type expr: CasADi SX or MX

        """
        assert self.flags['setup'] == False, 'Cannot call .set_alg after .setup.'
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (casadi.SX, casadi.MX, casadi.DM)), 'expr must be a casadi SX, MX or DM type, you have:{}'.format(type(expr))

        self.alg_list.append(entry(expr_name, expr = expr))


    def _convert2struct(self, var_dict):
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

        :param var_dict: Attributes that are configured with :py:func:`set_variable` (e.g. ``self._x``). These attributes are of type ``dict`` with the keys ``name`` and ``var``
        :type var_dict: dict
        """
        result_struct =  self.sv.sym_struct([
            entry(name, shape = var.shape) for var, name in zip(var_dict['var'], var_dict['name'])
        ])
        return result_struct

    def _substitute_struct_vars(self, var_dict_list, sym_struct_list, expr):
        """Helper function for :py:func:`setup`. Not part of the public API.
        This method is used to substitute the newly introduced structured variables with :py:func:`_convert2struct`
        into the expressions that define the model (e.g. ``_rhs``).

        **Why is this necessary?**

        For the symbolic variable type ``MX`` it is impossible to first create symbolic variables and then combine them into a structure (``struct_symMX``).
        We thus create symbolic variables, then create a structure holding similar variables and the substitute these newly introduced variables in all expressions.

        :param var_dict_list: List of attributes that are configured with :py:func:`set_variable` (e.g. ``self._x``). These attributes are of type ``dict`` with the keys ``name`` and ``var``
        :type var_dict_list: list
        :param sym_struct_list: List of the same attributes converted into structures with :py:func:`_convert2struct`.
        :type sym_struct_list: list
        :param expr: Casadi structured expr in which the variables from ``var_dict_list`` are substituted with those from ``sym_struct_list``.
        :type expr: struct_SX or struct_MX
        """
        assert len(var_dict_list)==len(sym_struct_list)

        subs = expr
        for var_dict, sym_struct in zip(var_dict_list, sym_struct_list):
            assert var_dict['name'] == sym_struct.keys()

            for var, name in zip(var_dict['var'], var_dict['name']):
                subs = substitute(subs, var, sym_struct[name])

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


    def setup(self):
        """Setup method must be called to finalize the modelling process.
        All required model variables must be declared.
        The right hand side expression for ``_x`` must have been set with :py:func:`set_rhs`.

        Sets default measurement function (state feedback) if :py:func:`set_meas` was not called.

        .. warning::

            After calling :py:func:`setup`, the model is locked and no further variables,
            expressions etc. can be set.

        :raises assertion: Definition of right hand side (rhs) is incomplete

        :return: None
        :rtype: None
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
        
        self.A_lin_expr = jacobian(self._rhs,self._x)
        self.B_lin_expr = jacobian(self._rhs,self._u)
        self.C_lin_expr = jacobian(self._y_expression,self._x)
        self.D_lin_expr = jacobian(self._y_expression,self._u)

        # Declare functions for the right hand side and the aux_expressions.
        self._rhs_fun = Function('rhs_fun',
                                 [_x, _u, _z, _tvp, _p, _w], [self._rhs],
                                 ["_x", "_u", "_z", "_tvp", "_p", "_w"], ["_rhs"])
        self._alg_fun = Function('alg_fun',
                                 [_x, _u, _z, _tvp, _p, _w], [self._alg],
                                 ["_x", "_u", "_z", "_tvp", "_p", "_w"], ["_alg"])
        self._aux_expression_fun = Function('aux_expression_fun',
                                            [_x, _u, _z, _tvp, _p], [self._aux_expression],
                                            ["_x", "_u", "_z", "_tvp", "_p"], ["_aux_expression"])
        self._meas_fun = Function('meas_fun',
                                  [_x, _u, _z, _tvp, _p, _v], [self._y_expression],
                                  ["_x", "_u", "_z", "_tvp", "_p", "_v"], ["_y_expression"])
        self.A_fun = Function('A_fun',
                              [_x, _u, _z, _tvp, _p, _w],[self.A_lin_expr],
                              ["_x", "_u", "_z", "_tvp", "_p", "_w"],["A_lin_expr"])
        self.B_fun = Function('B_fun',
                              [_x, _u, _z, _tvp, _p, _w],[self.B_lin_expr],
                              ["_x", "_u", "_z", "_tvp", "_p", "_w"],["B_lin_expr"])
        self.C_fun = Function('C_fun',
                              [_x, _u, _z, _tvp, _p, _v],[self.C_lin_expr],
                              ["_x", "_u", "_z", "_tvp", "_p", "_v"],["C_lin_expr"])
        self.D_fun = Function('D_fun',
                              [_x, _u, _z, _tvp, _p, _v],[self.D_lin_expr],
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

    def linearize(self, xss=None, uss=None, tvp0 = None, p0 = None):
        """Linearize the non-linear model to linear model.
        
        This method uses the taylor expansion series to linearize non-linear model to linear model at the specified 
        set points. Linearized model retains the same variable names for all states, inputs with respect to the original model. 
        The non-linear model equation this method can solve is as follows:
        
            .. math::
                \\dot{x} = f(x,u)
                
        The above model is linearized around steady state set point :math:`x_{ss}` and steady state input :math:`u_{ss}`
        
        .. math::
            \\frac{\\partial f}{\\partial x}|_{x_{ss}} = 0 \\\\
            \\frac{\\partial f}{\\partial u}|_{u_{ss}} = 0
                
        The linearized model is as follows:
            
            .. math::
                \\Delta\\dot{x} = A \\Delta x + B \\Delta u
                
        Similarly it can be extended to discrete time systems. Since the linearized model has only rate of change input and state. The names are appended with 'del' to differentiate 
        from the original model. This can be seen in the above model definition. Therefore, the solution of the lqr will be ``u`` and its corresponding ``x``. In order to fetch :math:`\\Delta u` 
        and :math:`\\Delta x`, setpoints has to be subtracted from the solution of lqr.
                
        :param xss: Steady state state
        :type xss: numpy.ndarray
        
        :param uss: Steady state input
        :type uss: numpy.ndarray
        
        :return: Linearized Model
        :rtype: LinearModel 
        

        """
        #Check whether model setup is done
        assert self.flags['setup'] == True, 'Run this function after original model is setup'
        assert self.z.size == 0, 'Linearization around steady state is not supported for DAEs'    
        
        A,B,C,D = self.get_linear_system_matrices(xss,uss, tvp=tvp0, p=p0)
        
        # Check if A,B,C,D are constant or expressions
        all_constant = np.alltrue(
            [isinstance(A, np.ndarray), isinstance(B, np.ndarray), isinstance(C, np.ndarray), isinstance(D, np.ndarray)]
        )
        
        if all_constant:
            # If all are constant, linear model is initialized
            linearizedModel = LinearModel(self.model_type,self.symvar_type)
        else:
            # If not, LTV model is initialized
            raise NotImplementedError('LTV models are not yet implemented.')


        # Create new variables for linearized model
        self._transfer_variables(self, linearizedModel)
        n_x = self.n_x
        n_u = self.n_u
        
        # Check for trivial measurement equation
        if C.shape == (n_x,n_x) and (C == np.eye(n_x)).all():
            C = None
        if D.shape == (n_x,n_u) and (D == np.zeros((n_x,n_u))).all():
            D = None
        
        # Setup linearized model
        linearizedModel.setup(A,B,C,D)
        
        return linearizedModel

    
    @staticmethod
    def _transfer_variables(old_model, new_model, transfer=['_x', '_u', '_tvp', '_p']):
        """Private and static method to transfer variables from old model to new model.
        This is used in :meth:`Model.linearize` and :meth:`Model.discretize`.

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
        
    
    def get_linear_system_matrices(self, xss=None, uss=None, z=None, tvp=None,p=None):
        """
        Returns the matrix quadrupel :math:`(A,B,C,D)` of the linearized system around the operating point (``xss,uss,z,tvp,p,w,v``).
        All arguments are optional in which case the matrices might still be symbolic. 
        If the matrices are not symbolic, they are returned as numpy arrays.
        
        :param xss: Steady state state (optional)
        :type xss: numpy.ndarray
        :param uss: Steady state input (optional)
        :type uss: numpy.ndarray
        :param z: steady state algebraic states (optional)
        :type z: numpy.ndarray
        :param tvp: time varying parameters set point (optional)
        :type tvp: numpy.ndarray
        :param p: parameters set point (optional)
        :type p: numpy.ndarray
        
        :return: A - State matrix
        :rtype: numpy.ndarray / CasADi SX
        :return: B - Input matrix
        :rtype: numpy.ndarray / CasADi SX
        :return: C - Output matrix
        :rtype: numpy.ndarray / CasADi SX
        :return: D - Feedforward matrix
        :rtype: numpy.ndarray / CasADi SX
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
            A = DM(A).full()
        B = self.B_fun(xss, uss, z, tvp, p, w)
        if B.is_constant():
            B = DM(B).full()
        C = self.C_fun(xss, uss, z, tvp, p, v)
        if C.is_constant():
            C = DM(C).full()
        D = self.D_fun(xss, uss, z, tvp, p, v)
        if D.is_constant():
            D = DM(D).full()
        return A,B,C,D

class LinearModel(Model):
    """The **do-mpc** LinearModel class. This class is inherited from **do-mpc** model class. 
    This class holds the full model description and is at the core of
    :py:class:`do_mpc.simulator.Simulator`, :py:class:`do_mpc.controller.MPC`, :py:class:`do_mpc.controller.LQR` and :py:class:`do_mpc.estimator.Estimator`.
    This class can be used to define the linear time invariant models in both
    continuous and discrete time.
    The :py:class:`LinearModel` class is created with setting the ``model_type`` (continuous or discrete).
    
    A ``continous`` linear model consists of an underlying ordinary differential equation (ODE)
    
    .. math::
        
        \\dot{x}(t) &= Ax(t)+Bu(t),\\\\
            y &= Cx(t)+Du(t)
            
    whereas a ``discrete`` linear model consists of a difference equation.
        
    .. math::
             
        x_{k+1} &= Ax_k+Bu_k,\\\\
       y_k &= Cx_k+Du_k
       
    The **do-mpc** linear model can be initiated with ``SX`` variable type.
    
    .. note::

        The option ``symvar_type`` will be inherited to all derived classes (e.g. :py:class:`do_mpc.simulator.Simulator`,
        :py:class:`do_mpc.controller.MPC` and :py:class:`do_mpc.estimator.Estimator`).
        All symbolic variables in these classes will be chosen respectively.
        
    **Configuration and setup:**
    Configuring and setting up the :py:class:`LinearModel` involves the following steps:
    
    Model can be setup in two different ways. The first method is as follows:
    
        1. Use :py:func:`set_variable` to introduce new variables to the linear model.
    
        2. Optionally introduce "auxiliary" expressions as functions of the previously defined variables with :py:func:`set_expression`. The expressions can be used for monitoring or be reused as constraints, the cost function etc.
    
        3. Optionally introduce measurement equations with :py:func:`set_meas`. The syntax is identical to :py:func:`set_expression`. By default state-feedback is assumed.
    
        4. Define the right-hand-side of the `discrete` or `continuous` model as a function of the previously defined variables with :py:func:`set_rhs`. This method must be called once for each introduced state.
    
        5. Call :py:func:`setup` to finalize the :py:class:`LinearModel`. No further changes are possible afterwards.
        
    The second method is as follows:
        
        1. Use :py:func:`set_variable` to introduce new variables to the linear model.
        
        2. Optionally introduce "auxiliary" expressions as functions of the previously defined variables with :py:func:`set_expression`. The expressions can be used for monitoring or be reused as constraints, the cost function etc.
        
        3. Call :py:func:`setup` and pass the system dynamics matrices as arguments instead of setting up the right hand side equations and measurement equations to finalize the :py:class:`LinearModel`. No further changes are possible afterwards.
        
    .. note::

        All introduced model variables are accessible as **Attributes** of the :py:class:`Model`.
        Use these attributes to query to variables, e.g. to form the cost function in a seperate file for the MPC configuration.
        
    :param model_type: Set if the model is ``discrete`` or ``continuous``.
    :type model_type: str
    :param symvar_type: Set if the model is configured with CasADi ``SX`` variables (default).
    :type symvar_type: str

    :raises assertion: model_type must be string
    :raises assertion: model_type must be either discrete or continuous

    .. automethod:: __getitem__
    """
    def __init__(self, model_type=None, symvar_type='SX'):
        super().__init__(model_type, symvar_type)
        if symvar_type == 'MX':
            raise ValueError("class LinearModel can be initialized only with SX variable.")


    @property
    def sys_A(self):
        """State matrix.
        This property provides the state matrix in the numerical array format. Accessible only after model is setup.
        
        :return: A - State matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._A
    
    @property
    def sys_B(self):
        """Input matrix.
        This property provides the input matrix in the numerical array format. Accessible only after model is setup.
        
        :return: B - Input matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._B
    
    @property
    def sys_C(self):
        """Output matrix.
        This property provides the output matrix in the numerical array format. Accessible only after model is setup.
        
        :return: C - Output matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._C
    
    @property
    def sys_D(self):
        """Feedforward matrix.
        This property provides the feedforward matrix in the numerical array format. Accessible only after model is setup.
        
        :return: D - Feedforward matrix
        :rtype: numpy.ndarray
        """
        assert self.flags['setup'] == True, 'Attributes are available after the model is setup.'
        return self._D


    def set_rhs(self, name, rhs):
        """
        Checks if the right-hand-side function is linear and calls :meth:`Model.set_rhs`.
        
        :param name: Reference to previously introduced state names (with :py:func:`LinearModel.set_variable`)
        :type name: sting
        :param rhs: CasADi SX function depending on ``_x``, ``_u``, ``_tvp``, ``_p``.
        :type rhs: casADi SX
        """
        # Check if expression is linear
        if evalf(jacobian(rhs, vertcat(self.x, self.u))).is_constant():
            super(LinearModel, self).set_rhs(name, rhs, process_noise=True)
        else:
            raise ValueError("Given rhs is not linear.")

    def set_meas(self, name, meas):
        """
        Checks if the measurement function is linear and calls :meth:`Model.set_meas`.
        
        :param name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type name: sting
        :param meas: CasADi SX function depending on ``_x``, ``_u``, ``_tvp``, ``_p``.
        :type meas: casADi SX
        """
        # Check if expression is linear
        if evalf(jacobian(meas, vertcat(self.x, self.u))).is_constant():
            super(LinearModel, self).set_meas(name, meas, meas_noise=True)

    def set_alg(self, expr_name, expr, *args, **kwargs):
        """
        .. warning::

            This method is not supported for linear models.
        """
        raise NotImplementedError('Algebraic variables are not supported for linear models.')
        
    def setup(self,A=None, B=None,C=None,D=None):
        """Setup method must be called to finalize the modelling process.
        All required model variables must be declared.
        The right hand side expression for ``_x`` can be set with :py:func:`set_rhs` or can be set by passing the state matrix and input matrix in :py:func:`setup`.

        Sets default measurement function (state feedback) if :py:func:`set_meas` was not called or output matrix, feedforward matrix are not passed in :py:func:`setup`.

        .. warning::

            After calling :py:func:`setup`, the model is locked and no further variables,
            expressions etc. can be set.

        :raises assertion: Definition of right hand side (rhs) is incomplete
        
        :param name: A - State matrix (optional)
        :type name: numpy.ndarray
        :param name: B - Input matrix (optional)
        :type name: numpy.ndarray
        :param name: C - Output matrix (optional)
        :type name: numpy.ndarray
        :param name: D - Feedforward matrix (optional)
        :type name: numpy.ndarray

        :return: None
        :rtype: None
        """
        if not isinstance(A, (np.ndarray, type(None))):
            raise ValueError('A must be a numpy array or None')
        if not isinstance(B, (np.ndarray, type(None))):
            raise ValueError('B must be a numpy array or None')
        if not isinstance(C, (np.ndarray, type(None))):
            raise ValueError('C must be a numpy array or None')
        if not isinstance(D, (np.ndarray, type(None))):
            raise ValueError('D must be a numpy array or None')
    


        # Three use cases:
        # 1. C / D are given -> Create measurement function
        # 2. set_meas has been called -> Measurement function already exists
        # 3. Neither C / D nor set_meas -> No measurement super().setup() creates default measurement function (state feedback)        
        y_meas = None
        if not isinstance(C, type(None)):
            y_meas = C @ self.x.cat
        if not isinstance(D, type(None)):
            y_meas = y_meas + D @ self.u.cat
        if not isinstance(y_meas, type(None)):
            n_y = y_meas.shape[0]
            self.set_meas('y', y_meas)

        n_x = self.x.size
        n_u = self.u.size

        # Create x_next from A and B matrices (if available)
        x_next = None
        if isinstance(A, np.ndarray):
            if A.shape != (n_x, n_x):
                raise ValueError('A must be a square matrix with size n_x x n_x. You have A.shape={}'.format(A.shape))
            else:
                x_next = A @ self.x.cat 
        if isinstance(B, np.ndarray):
            if B.shape != (n_x, n_u):
                raise ValueError('B must be a matrix with size n_x x n_u. You have B.shape={}'.format(B.shape))
            else:
                x_next = x_next + B @ self.u.cat    

        # Set the rhs of the states (if x_next exists)
        if not isinstance(x_next, type(None)):
            for name in self.x.keys():
                ind = self.x.f[name]
                self.set_rhs(name, x_next[ind])


        super(LinearModel, self).setup()

        # Create A,B,C,D matrices (they are not necessarily given) and write them to the class.
        A,B,C,D = self.get_linear_system_matrices()
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        
    
    def discretize(self, t_sample = 0, conv_method = 'zoh'):
        """Converts continuous time to discrete time system.
        
        This method utilizes the exisiting function in scipy library called ``cont2discrete`` to convert continuous time to discrete time system.This method 
        allows the user to specify the type of discretization. For more details about the function `click here`_ .
         
        .. _`click here`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.cont2discrete.html
            
        where :math:`A_{\\text{discrete}}` and :math:`B_{\\text{discrete}}` are the discrete state matrix and input matrix repectively and :math:`t_{\\text{sample}}`
        is the sampling time.
        
        .. warning::
            sampling time is zero when not specified or not required
        :param t_sample: Sampling time (default - ``0``)
        :type t_sample: Int or float
        
        :param conv_method: Method of discretization - Five different methods can be applied. (default -`` zoh``)
        :type conv_method: String
        
        :return: model
        :rtype: LinearModel
        """
        assert self.flags['setup'] == True, 'This method can be accessed only after the model is setup using LinearModel.setup().'
        assert self.model_type == 'continuous', 'Given model is already discrete.'
        warnings.warn('sampling time is {}'.format(t_sample))
        
        A, B, C, D, t = cont2discrete((self.sys_A,self.sys_B,self.sys_C,self.sys_D), t_sample, conv_method)
        
        discreteModel = LinearModel('discrete')

        # Create new variables for linearized model
        self._transfer_variables(self, discreteModel)

        # Setup linearized model
        discreteModel.setup(A,B)
        
        return discreteModel 
        
        
    def get_steady_state(self,xss = None,uss = None):
        """Calculates steady states for the given input or states.
        
        This method calculates steady states of a discrete system for the given steady state input and vice versa.
        The mathematical formulation can be described as 
            
            .. math::
                x_{ss} = (I-A)^{-1}Bu_{ss}\\\\
                
               or\\\\
                
                u_{ss} = B^{-1}(I-A)x_{ss}
        
        :return: Steady state
        :rtype: numpy.ndarray
        
        :return: Steady state input
        :rtype: numpy.ndarray
        """
        #Check whether the model is linear and setup
        assert self.flags['setup'] == True, 'Model is not setup. Please run model.setup() fun to calculate steady state.'
        assert self.model_type == 'discrete', 'Please convert the system to discrete using model.continuous_2_discrete().'
        I = np.identity(np.shape(self.sys_A)[0])
        
        #Calculation of steady state
        if np.all(xss) == None:
            assert np.all(uss) != None and isinstance(uss,np.ndarray), 'Provide either steady state states or steady state inputs.'
            self.xss = np.linalg.inv(I-self.sys_A)@self.sys_B@uss
            self.uss = uss
            return self.xss
        elif np.all(uss) == None and np.shape(self.sys_B)[0] != np.shape(self.sys_B)[1]:
            assert np.all(xss) != None and isinstance(xss,np.ndarray), 'Provide either steady state states or steady state inputs.'
            self.uss = np.linalg.pinv(self.sys_B)@(I-self.sys_A)@xss
            self.xss = xss
            return self.uss
        elif np.all(uss) == None and np.shape(self.sys_B)[0] == np.shape(self.sys_B)[1]:
            assert np.all(xss) != None and isinstance(xss,np.ndarray), 'Provide either steady state states or steady state inputs.'
            self.uss = np.linalg.inv(self.sys_B)@(I-self.sys_A)@xss
            self.xss = xss
            return self.uss
        

