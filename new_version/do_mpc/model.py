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


class model:
    def __init__(self, model_type=None):
        """Initiate the do_mpc model class. This class holds the full model description and is at the core of
        simulator, optimizer and estimator. The model class is created with setting the model_type (continuous or discrete).
        The final model is created with the method .setup_model().
        Prior to that call the following variables have to be declared (with .set_variable):
        - states (_x)
        - inputs (_u)
        - algebraic (_z)
        - parameter (_p)
        - timevarying_parameter (_tvp)
        As well as auxiliary expressions (with .set_expression)
        and finally the right hand side (rhs) of the ODE with .set_rhs().

        :param model_type: Set if the model is discrete or continuous
        :type var_type: string

        :raises assertion: var_type must be string
        :raises assertion: var_name must be string
        :raises assertion: shape must be tuple or int

        :return: None
        :rtype: None
        """
        assert isinstance(model_type, str), 'model_type must be string, you have: {}'.format(type(model_type))
        assert model_type in ['discrete', 'continuous'], 'model_type must be either discrete or continuous, you have: {}'.format(model_type)

        self.model_type = model_type

        # Initilize lists for variables, expressions and rhs to be added to the model
        self.var_list = {
            '_x': [],
            '_u': [{'var_name': 'default', 'value':SX.sym('default',(0,0))}],
            '_z': [{'var_name': 'default', 'value':SX.sym('default',(0,0))}],
            '_p': [{'var_name': 'default', 'value':SX.sym('default',(0,0))}],
            '_tvp': [{'var_name': 'default', 'value':SX.sym('default',(0,0))}],
        }
        self.expr_list = []
        self.rhs_list = []


    def set_variable(self, var_type, var_name, shape=(1,1)):
        """Introduce new variables to the model class. Define variable type, name and shape (optional).

        :param var_type: Declare the type of the variable. The following types are valid (long or short name is possible):
        - states (_x)
        - inputs (_u)
        - algebraic (_z)
        - parameter (_p)
        - timevarying_parameter (_tvp)
        :type var_type: string
        :param var_name: Set a user-defined name for the parameter. The names are reused throughout do_mpc.
        :type var_type: string
        :param shape: Shape of the current variable (optional), default is scalar variable.
        :type var_type: int or tuple of length 2.

        :raises assertion: var_type must be string
        :raises assertion: var_name must be string
        :raises assertion: shape must be tuple or int

        :return: Returns the newly created symbolic variable. Variables can be used to create aux expressions.
        :rtype: casadi.SX
        """
        assert isinstance(var_type, str), 'var_type must be str, you have: {}'.format(type(var_type))
        assert isinstance(var_name, str), 'var_name must be str, you have: {}'.format(type(var_name))
        assert isinstance(shape, (tuple,int)), 'shape must be tuple or int, you have: {}'.format(type(shape))

        var = SX.sym(var_name,shape)
        if var_type in ['states', '_x']:
            self.var_list['_x'].extend([{'var_name': var_name, 'value':var}])
        elif var_type in ['inputs', '_u']:
            self.var_list['_u'].extend([{'var_name': var_name, 'value':var}])
        elif var_type in ['algebraic', '_z']:
            self.var_list['_z'].extend([{'var_name': var_name, 'value':var}])
        elif var_type in ['parameter', '_p']:
            self.var_list['_p'].extend([{'var_name': var_name, 'value':var}])
        elif var_type in ['timevarying_parameter', '_tvp']:
            self.var_list['_tvp'].extend([{'var_name': var_name, 'value':var}])
        else:
            warning('Trying to set non-existing variable var_type: {} with var_name {}'.format(var_type, var_name))

        return var

    def set_expression(self, expr_name, expr):
        """Introduce new expression to the model class. Expressions are not required but can be used
        to extract further information from the model. They can also be use for the objective function or constraints.
        Expressions must be formulated with respect to _x, _u, _z, _tvp, _p.

        :param expr_name: Arbitrary name for the given expression. Names are used for key word indexing.
        :type expr_name: string
        :param expr: CasADi SX or MX function depending on _x, _u, _z, _tvp, _p.
        :type expr: CasADi SX or MX

        :raises assertion: expr_name must be str
        :raises assertion: expr must be a casadi SX or MX type

        :return: Returns the newly created expression. Expression can be used e.g. for the RHS.
        :rtype: casadi.SX
        """
        assert isinstance(expr_name, str), 'expr_name must be str, you have: {}'.format(type(expr_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))
        self.expr_list.extend([{'expr_name': expr_name, 'expr': expr}])

        return expr

    def set_rhs(self, var_name, expr):
        """Formulate the right hand side (rhs) of the ODE. Each defined state variable must have a respective equation (of matching dimension)
        for the rhs. Match the rhs with the state by choosing the corresponding names.
        RHS must be formulated with respect to _x, _u, _z, _tvp, _p.

        Example:
        tank_level = model.set_variable('states', 'tank_level')
        tank_temp = model.set_variable('states', 'tank_temp')

        tank_level_next = 0.5*tank_level
        tank_temp_next = ...

        model.set_rhs('tank_level', tank_level_next)
        model.set_rhs('tank_temp', tank_temp_next)

        :param var_name: Name of any of the previously defined states with: model.set_variable('states', [NAME])
        :type var_name: string
        :param expr: CasADi SX or MX function depending on _x, _u, _z, _tvp, _p.
        :type expr: CasADi SX or MX

        :raises assertion: var_name must be str
        :raises assertion: expr must be a casadi SX or MX type
        :raises assertion: var_name must refer to the previously defined states

        :return: None
        :rtype: None
        """
        assert isinstance(var_name, str), 'var_name must be str, you have: {}'.format(type(var_name))
        assert isinstance(expr, (casadi.SX, casadi.MX)), 'expr must be a casadi SX or MX type, you have:{}'.format(type(expr))
        _x_names = [_x_dict_i['var_name']  for _x_dict_i in self.var_list['_x']]
        assert var_name in _x_names, 'var_name must refer to the previously defined states ({}). You have: {}'.format(_x_names, var_name)
        self.rhs_list.extend([{'var_name': var_name, 'expr': expr}])

    def get_variables(self):
        # TODO: Add docstring.
        try:
            return self._x, self._u, self._z, self._tvp, self._p, self._aux_expression
        except:
            print('get_variables could not be called. Call setup_model first.')
            raise


    def setup_model(self):
        """Setup method must be called to finalize the modelling process.
        All model variables _x, _u, _z, _tvp, _p must be declared.
        The right hand side expression for _x must have been set.

        :raises assertion: Definition of right hand side (rhs) is incomplete

        :return: None
        :rtype: None
        """

        # Write self._x, self._u, self._z, self._tvp, self.p with the respective struct_symSX structures.
        # Use the previously defined SX.sym variables to declare shape and symbolic variable.
        for var_type_i, var_list_i in self.var_list.items():
            struct_i  = struct_symSX([
                entry(var_ik['var_name'], sym=var_ik['value']) for var_ik in var_list_i
            ])
            setattr(self, var_type_i, struct_i)

        # Write self._aux_expression.
        # Use the previously defined SX.sym variables to declare shape and symbolic variable.
        self._aux_expression = struct_SX([
            entry(expr_i['expr_name'], expr=expr_i['expr']) for expr_i in self.expr_list
        ])

        # Create mutable struct_SX with identical structure as self._x to hold the right hand side.
        self._rhs = struct_SX(self._x)

        # Set the expressions in self._rhs with the previously defined SX.sym variables.
        # Check if an expression is set for every state of the system.
        _x_names = set([_x_dict_i['var_name']  for _x_dict_i in self.var_list['_x']])
        for rhs_i in self.rhs_list:
            self._rhs[rhs_i['var_name']] = rhs_i['expr']
            _x_names -= set([rhs_i['var_name']])
        assert len(_x_names) == 0, 'Definition of right hand side (rhs) is incomplete. Missing: {}. Use: set_rhs to define expressions.'.format(_x_names)

        # Declare functions for the right hand side and the aux_expressions.
        _x, _u, _z, _tvp, _p, _aux = self.get_variables()
        self._rhs_fun = Function('rhs_fun', [_x, _u, _z, _tvp, _p], [self._rhs])
        self._aux_expression_fun = Function('aux_expression_fun', [_x, _u, _z, _tvp, _p], [_aux])

        # Create and store some information about the model regarding number of variables for
        # x, _u, _z, _tvp, _p, _aux
        self.n_x = self._x.shape[0]
        self.n_u = self._u.shape[0]
        self.n_z = self._z.shape[0]
        self.n_tvp = self._tvp.shape[0]
        self.n_p = self._p.shape[0]
        self.n_aux = self._aux_expression.shape[0]
