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
import do_mpc.data


class optimizer:
    def __init__(self, model):
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
        self._t0 = np.array([0])


    def set_param(self, n_horizon=None, n_robust=None, open_loop=None, t_step=None, state_discretization=None):
        # TODO: Add docstring.
        if n_horizon:
            self.n_horizon = n_horizon
        if n_robust:
            self.n_robust = n_robust
        if open_loop:
            self.open_loop = open_loop
        if t_step:
            self.t_step = t_step
        if state_discretization:
            self.state_discretization = state_discretization

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

    def set_tvp_fun(self, tvp_fun):
        # TODO: Add docstring.
        # TODO: Implement checks regarding n_horizon, n_tvp, etc.
        None

    def set_uncertainty_values(self, uncertainty_values):
        # TODO: Check correct format.
        assert uncertainty_values.shape[0] == self.model.n_p, 'asdf'
        assert uncertainty_values.ndim == 2, 'asdf'
        self.uncertainty_values = uncertainty_values

    def check_validity(self):
        None

    def setup_discretization(self):
        _x, _u, _z, _tvp, _p, _aux = self.model.get_variables()
        if self.state_discretization:
            self.x_next_fun = Function('x_next_fun', [_x, _u, _z, _tvp, _p], [self.model._rhs])

    def setup_scenario_tree(self):
        """
        -----------------------------------------------------------------------------------
        Build the scenario tree given the possible values of the uncertain parmeters
        The strategy to build the tree is by default a combination of all the possible values
        This strategy can be modified by changing the code below
        -----------------------------------------------------------------------------------
        """
        # TODO: This function can be written nicer
        uncertainty_values = self.uncertainty_values
        n_p = self.model.n_p
        # Initialize some auxiliary variables
        # pdb.set_trace()
        current_scenario = np.resize(np.array([], dtype=int), n_p)
        p_scenario_index = np.resize(np.array([], dtype=int), n_p)
        p_scenario = np.resize(np.array([]), n_p)
        number_values_per_uncertainty = np.resize(np.array([], dtype=int), n_p)
        k = 1
        # Get the number of different values of each parameter
        for ii in range(n_p):
            number_values_per_uncertainty[ii] = uncertainty_values[ii].size
            current_scenario[ii] = 0
        while (current_scenario != number_values_per_uncertainty - 1).any():
            for index in range(n_p - 1, -1, -1):
                if current_scenario[index] + \
                        1 < number_values_per_uncertainty[index]:
                    # If it is no the last element increase it and break the for
                    # loop
                    current_scenario[index] += np.array([1])
                    break
                else:
                    current_scenario[index] = np.array([0])
            # Add the current scenario to the variable p_scenario
            p_scenario_index = np.vstack((p_scenario_index, current_scenario))

        # Initialize the vector with the real values
        p_scenario = p_scenario_index * 1.0
        for jj in range(len(p_scenario_index)):
            for ii in range(n_p):
                p_scenario[jj, ii] = uncertainty_values[ii][p_scenario_index[jj, ii]]

        return p_scenario

    def setup_nlp(self):
        self.check_validity()

        self.setup_discretization()
        p_scenario = self.setup_scenario_tree()
        # Create struct for optimization variables:
        self.opt_x = opt_x = struct_symSX([
            entry('_x', repeat=self.n_horizon+1, struct=self.model._x),
            entry('_z', repeat=self.n_horizon, struct=self.model._z),
            entry('_u', repeat=self.n_horizon, struct=self.model._u),
        ])

        # Number of optimization variables:
        self.n_x_optim = opt_x.shape[0]

        # Create struct for optimization parameters:
        self.opt_p = opt_p = struct_symSX([
            entry('_x0', struct=self.model._x),
            entry('_tvp',    repeat=self.n_horizon, struct=self.model._tvp),
            entry('_p', struct=self.model._p),
            entry('_u_prev', struct=self.model._u)
        ])

        # self.mpc_obj_aux = struct_MX(struct_symMX([
        #     entry('nl_cons', repeat=self.n_horizon, struct=self.nl_cons)
        # ]))

        self.lb_opt_x = opt_x(-np.inf)
        self.ub_opt_x = opt_x(np.inf)

        # Initialize objective function and constraints
        obj = 0
        cons = []
        cons_lb = []
        cons_ub = []

        # Initial condition:
        cons.append(opt_x['_x', 0]-opt_p['_x0'])

        cons_lb.append(np.zeros((self.model.n_x, 1)))
        cons_ub.append(np.zeros((self.model.n_x, 1)))
        # Note:
        # X = [x_0, x_1, ... , x_(N+1)]         -> n_horizon+1 elements
        # U = [u_0, u_1, ... , u_N]             -> n_horizon elements

        for k in range(self.n_horizon):
            # Add constraints for state equation:
            x_next = self.x_next_fun(opt_x['_x', k], opt_x['_u', k], opt_x['_z', k], opt_p['_tvp', k], opt_p['_p'])

            cons.append(x_next-opt_x['_x', k+1])
            cons_lb.append(np.zeros((self.model.n_x, 1)))
            cons_ub.append(np.zeros((self.model.n_x, 1)))

            nl_cons_k = self._nl_cons_fun(opt_x['_x', k], opt_x['_u', k], opt_x['_z', k], opt_p['_tvp', k], opt_p['_p'])
            cons.append(nl_cons_k)
            cons_lb.append(self._nl_cons_lb)
            cons_ub.append(self._nl_cons_ub)

            obj += self.lterm_fun(opt_x['_x', k], opt_x['_u', k], opt_x['_z', k], opt_p['_tvp', k], opt_p['_p'])

            # U regularization:
            if k == 0:
                obj += self.rterm_factor.cat.T@((opt_x['_u', 0]-opt_p['_u_prev'])**2)
            else:
                obj += self.rterm_factor.cat.T@((opt_x['_u', k]-opt_x['_u', k-1])**2)

            self.lb_opt_x['_x', k] = self._x_lb
            self.ub_opt_x['_x', k] = self._x_ub

            self.lb_opt_x['_u', k] = self._u_lb
            self.ub_opt_x['_u', k] = self._u_ub

        obj += self.mterm_fun(opt_x['_x', self.n_horizon])

        self.lb_opt_x['_x', self.n_horizon] = self._x_terminal_lb
        self.ub_opt_x['_x', self.n_horizon] = self._x_terminal_ub

        cons = vertcat(*cons)
        self.cons_lb = vertcat(*cons_lb)
        self.cons_ub = vertcat(*cons_ub)

        # Create casadi optimization object:
        optim_opts = {}
        optim_opts["expand"] = False
        optim_opts["ipopt.linear_solver"] = 'ma27'
        nlp = {'x': vertcat(opt_x), 'f': obj, 'g': cons, 'p': vertcat(opt_p)}
        self.S = nlpsol('S', 'ipopt', nlp, optim_opts)

        # Create copies of these structures with numerical values (all zero):
        self.opt_x_num = self.opt_x(0)
        self.opt_p_num = self.opt_p(0)

    def solve(self):
        r = self.S(x0=self.opt_x_num, lbx=self.lb_opt_x, ubx=self.ub_opt_x,  ubg=self.cons_ub, lbg=self.cons_lb, p=self.opt_p_num)
        self.opt_x_num = self.opt_x(r['x'])
        # Values of lagrange multipliers:
        self.lam_g_num = r['lam_g']
        self.solver_stats = self.S.stats()
