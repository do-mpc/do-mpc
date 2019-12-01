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
import do_mpc.data
from do_mpc import backend_optimizer


class optimizer(backend_optimizer):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.data = do_mpc.data.optimizer_data(model)

        self._x_lb = model._x(-np.inf)
        self._x_ub = model._x(np.inf)

        self._x_terminal_lb = model._x(-np.inf)
        self._x_terminal_ub = model._x(np.inf)

        self._u_lb = model._u(-np.inf)
        self._u_ub = model._u(np.inf)

        self._x_scaling = model._x(1)
        self._u_scaling = model._u(1)

        self._x0 = model._x(0)
        self._u0 = model._u(0)
        self._z0 = model._z(0)
        self._t0 = np.array([0])

        # Parameters that can be set for the optimizer:
        self.data_fields = [
            'n_horizon',
            'n_robust',
            'open_loop',
            't_step',
            'state_discretization',
            'collocation_type',
            'collocation_deg',
            'collocation_ni',
        ]

        # Default Parameters:
        self.n_robust = 0
        self.collocation_type = 'radau'
        self.collocation_deg = 2
        self.collocation_ni = 1
        self.open_loop = False

    def set_param(self, **kwargs):
        """[Summary]

        :param integration_tool: , defaults to None
        :type [ParamName]: [ParamType](, optional)
        :raises [ErrorType]: [ErrorDescription]
        :return: [ReturnDescription]
        :rtype: [ReturnType]
        """
        for key, value in kwargs.items():
            if not (key in self.data_fields):
                print('Warning: Key {} does not exist for optimizer.'.format(key))
            setattr(self, key, value)

    def set_nl_cons(self, **kwargs):
        # TODO: Make sure kwargs are passed correctly.
        self._nl_cons = struct_SX([
            entry(name, expr=expr) for name, expr in kwargs.items()
        ])
        _x, _u, _z, _tvp, _p, _aux = self.model.get_variables()
        self._nl_cons_fun = Function('nl_cons_fun', [_x, _u, _z, _tvp, _p], [self._nl_cons])
        self._nl_cons_ub = self._nl_cons(0)
        self._nl_cons_lb = self._nl_cons(-np.inf)

    def set_objective(self, mterm=None, lterm=None):
        # TODO: Add docstring
        _x, _u, _z, _tvp, _p, _aux = self.model.get_variables()

        # TODO: Check if this is only a function of x
        self.mterm = mterm
        # TODO: This function should be evaluated with scaled variables.
        self.mterm_fun = Function('mterm', [_x], [mterm])

        self.lterm = lterm
        self.lterm_fun = Function('lterm', [_x, _u, _z, _tvp, _p], [lterm])

    def get_rterm(self):
        # TODO: Check if called before setup_nlp
        self.rterm_factor = self.model._u(0)
        return self.rterm_factor

    def get_tvp_template(self):
        tvp_template = struct_symSX([
            entry('_tvp', repeat=self.n_horizon, struct=self.model._tvp)
        ])
        return tvp_template(0)

    def set_tvp_fun(self,tvp_fun):
        assert self.get_tvp_template().labels() == tvp_fun(0).labels(), 'Incorrect output of tvp_fun. Use get_tvp_template to obtain the required structure.'
        self.tvp_fun = tvp_fun


    def get_p_template(self, n_combinations):
        """ Low level API method to set user defined scenarios for robust MPC but defining an arbitrary number
        of combinations for the parameters defined in the model. The method returns a structured object which is
        initialized with all zeros. Use this object to define value of the parameters for an arbitrary number of scenarios (defined by n_scenarios).

        This structure (with numerical values) should be used as the output of the p_fun function which is set to the class with .set_p_fun (see doc string).

        Use the combination of .get_p_template() and .set_p_template() as a more adaptable alternative to .set_uncertainty_values().

        Example:

        # in model definition:
        alpha = model.set_variable(var_type='_p', var_name='alpha')
        beta = model.set_variable(var_type='_p', var_name='beta')

        ...
        # in optimizer configuration:
        n_combinations = 3
        p_template = optimizer.get_p_template(n_combinations)
        p_template['_p',0] = np.array([1,1])
        p_template['_p',1] = np.array([0.9, 1.1])
        p_template['_p',2] = np.array([1.1, 0.9])

        def p_fun(t_now):
            return p_template

        optimizer.set_p_fun(p_fun)

        Note the nominal case is now:
        alpha = 1
        beta = 1
        which is determined by the order in the arrays above (first element is nominal).

        :param n_combinations: Define the number of combinations for the uncertain parameters for robust MPC.
        :type p_fun: int

        :return: None
        :rtype: None
        """
        self.n_combinations = n_combinations
        p_template = struct_symSX([
            entry('_p', repeat=n_combinations, struct=self.model._p)
        ])
        return p_template(0)


    def set_p_fun(self,p_fun):
        """ Low level API method to set user defined scenarios for robust MPC but defining an arbitrary number
        of combinations for the parameters defined in the model. The method takes as input a function, which MUST
        return a structured object, based on the defined parameters and the number of combinations.
        The defined function has time as a single input.

        Obtain this structured object first, by calling .get_p_template().

        Use the combination of .get_p_template() and .set_p_template() as a more adaptable alternative to .set_uncertainty_values().

        Example:

        # in model definition:
        alpha = model.set_variable(var_type='_p', var_name='alpha')
        beta = model.set_variable(var_type='_p', var_name='beta')

        ...
        # in optimizer configuration:
        n_combinations = 3
        p_template = optimizer.get_p_template(n_combinations)
        p_template['_p',0] = np.array([1,1])
        p_template['_p',1] = np.array([0.9, 1.1])
        p_template['_p',2] = np.array([1.1, 0.9])

        def p_fun(t_now):
            return p_template

        optimizer.set_p_fun(p_fun)

        Note the nominal case is now:
        alpha = 1
        beta = 1
        which is determined by the order in the arrays above (first element is nominal).

        :param p_fun: Function which returns a structure with numerical values. Must be the same structure as obtained from .get_p_template().
        Function must have a single input (time).
        :type p_fun: function

        :return: None
        :rtype: None
        """
        assert self.get_p_template(self.n_combinations).labels() == p_fun(0).labels(), 'Incorrect output of p_fun. Use get_p_template to obtain the required structure.'
        self.p_fun = p_fun

    def set_uncertainty_values(self, uncertainty_values):
        """ High-level API method to conveniently set all possible scenarios for multistage MPC, given a list of uncertainty values.
        This list must have the same number of elements as uncertain parameters in the model definition. The first element is the nominal case.
        Each list element can be an array or list of possible values for the respective parameter.
        Note that the order of elements determine the assignment.

        Example:

        # in model definition:
        alpha = model.set_variable(var_type='_p', var_name='alpha')
        beta = model.set_variable(var_type='_p', var_name='beta')
        ...
        # in optimizer configuration:
        alpha_var = np.array([1., 0.9, 1.1])
        beta_var = np.array([1., 1.05])
        optimizer.set_uncertainty_values([alpha_var, beta_var])

        Note the nominal case is now:
        alpha = 1
        beta = 1
        which is determined by the order in the arrays above (first element is nominal).

        :param uncertainty_values: List of lists / numpy arrays with the same number of elements as number of parameters in model.
        :type uncertainty_values: list

        :raises asssertion: uncertainty values must be of type list

        :return: None
        :rtype: None
        """
        assert isinstance(uncertainty_values, list), 'uncertainty values must be of type list, you have: {}'.format(type(uncertainty_values))

        p_scenario = list(itertools.product(*uncertainty_values))
        n_combinations = len(p_scenario)
        p_template = self.get_p_template(n_combinations)
        p_template['_p',:] = p_scenario
        def p_fun(t_now):
            return p_template

        self.set_p_fun(p_fun)


    def solve(self):
        r = self.S(x0=self.opt_x_num, lbx=self.lb_opt_x, ubx=self.ub_opt_x,  ubg=self.cons_ub, lbg=self.cons_lb, p=self.opt_p_num)
        self.opt_x_num = self.opt_x(r['x'])
        self.opt_g_num = r['g']
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.solver_stats = self.S.stats()
