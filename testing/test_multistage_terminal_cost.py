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

import sys
import unittest

import numpy as np

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc


def _solve(model, n_horizon, n_robust, x0):
    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=n_horizon,
        t_step=1.0,
        n_robust=n_robust,
        open_loop=False,
        store_full_solution=True,
        nlpsol_opts={'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0},
    )
    mpc.set_objective(mterm=model.x['x'] ** 2, lterm=0 * model.x['x'] ** 2)
    mpc.set_rterm(u=0)
    if model.n_p:
        mpc.set_uncertainty_values(d=np.array([1.0, -1.0]))
    mpc.setup()
    mpc.x0 = x0
    mpc.set_initial_guess()
    u0 = np.asarray(mpc.make_step(x0), dtype=float).reshape(-1)
    terminal = np.asarray(mpc.opt_x_num['_x', n_horizon, :, -1], dtype=float).reshape(-1)
    objective = float(mpc.solver_stats['iterations']['obj'][-1])
    return u0, terminal, objective


class TestMultistageTerminalCost(unittest.TestCase):
    def _disturbed_model(self):
        model = do_mpc.model.Model('discrete')
        x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        u = model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
        d = model.set_variable(var_type='_p', var_name='d', shape=(1, 1))
        model.set_rhs('x', x + u + d)
        model.setup()
        return model

    def test_branching_last_stage_prices_every_leaf(self):
        # Issue 477: horizon 1, disturbances +1 and -1, terminal cost x**2.
        # One shared input cannot zero both leaves. The equal-weight optimum is
        # u=-5, leaves +1 and -1, objective 1.
        u0, terminal, objective = _solve(self._disturbed_model(), n_horizon=1, n_robust=1, x0=np.array([5.0]))
        self.assertAlmostEqual(u0[0], -5.0, places=5)
        np.testing.assert_allclose(terminal[:2], [1.0, -1.0], atol=1e-5)
        self.assertAlmostEqual(objective, 1.0, places=5)

    def test_terminal_cost_reaches_zero_after_the_tree_stops_branching(self):
        # Robust horizon ends before the terminal step, so each leaf has its own input.
        u0, terminal, objective = _solve(self._disturbed_model(), n_horizon=2, n_robust=1, x0=np.array([5.0]))
        self.assertEqual(u0.shape, (1,))
        np.testing.assert_allclose(terminal[:2], [0.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(objective, 0.0, places=5)

    def test_nominal_terminal_cost(self):
        model = do_mpc.model.Model('discrete')
        x = model.set_variable(var_type='_x', var_name='x', shape=(1, 1))
        model.set_variable(var_type='_u', var_name='u', shape=(1, 1))
        model.set_rhs('x', x + model.u['u'])
        model.setup()
        u0, terminal, objective = _solve(model, n_horizon=1, n_robust=0, x0=np.array([5.0]))
        self.assertAlmostEqual(u0[0], -5.0, places=5)
        np.testing.assert_allclose(terminal[:1], [0.0], atol=1e-5)
        self.assertAlmostEqual(objective, 0.0, places=5)


if __name__ == '__main__':
    unittest.main()
