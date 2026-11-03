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

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc


def get_model():
    model = do_mpc.model.Model('discrete')
    x = model.set_variable('_x', 'x')
    u = model.set_variable('_u', 'u')
    model.set_meas('y_meas', x)
    model.set_expression('x_squared', x**2)
    model.set_rhs('x', x + u)
    model.setup()
    return model


class TestMPCSetObjective(unittest.TestCase):
    """Input validation of :py:meth:`do_mpc.controller.MPC.set_objective` (see #462)."""

    def setUp(self):
        self.model = get_model()
        self.mpc = do_mpc.controller.MPC(self.model)
        _x, _u, _z, _tvp, _p = self.model['x', 'u', 'z', 'tvp', 'p']
        self.u0, self.z0, self.tvp0, self.p0 = _u(0), _z(0), _tvp(0), _p(0)

    def lterm(self, mpc, x):
        return float(mpc.lterm_fun(x, self.u0, self.z0, self.tvp0, self.p0))

    def mterm(self, mpc, x):
        return float(mpc.mterm_fun(x, self.tvp0, self.p0))

    def test_aux_in_lterm(self):
        self.mpc.set_objective(lterm=self.model.aux['x_squared'], mterm=self.model.x['x']**2)
        self.assertEqual(self.lterm(self.mpc, 2), 4.0)
        self.assertEqual(self.mterm(self.mpc, 3), 9.0)

    def test_unset_terms_default_to_zero(self):
        self.mpc.set_objective(lterm=self.model.x['x']**2)
        self.assertEqual(self.mterm(self.mpc, 3), 0.0)

        mpc = do_mpc.controller.MPC(self.model)
        mpc.set_objective(mterm=self.model.x['x']**2)
        self.assertEqual(self.lterm(mpc, 3), 0.0)

    def test_measurement_variable_rejected_in_lterm(self):
        with self.assertRaisesRegex(Exception, 'lterm contains invalid symbolic variables'):
            self.mpc.set_objective(lterm=self.model.y['y_meas']**2, mterm=self.model.x['x']**2)

    def test_measurement_variable_rejected_in_mterm(self):
        with self.assertRaisesRegex(Exception, 'mterm contains invalid symbolic variables'):
            self.mpc.set_objective(lterm=self.model.x['x']**2, mterm=self.model.y['y_meas']**2)

    def test_input_rejected_in_mterm(self):
        with self.assertRaisesRegex(Exception, 'mterm contains invalid symbolic variables'):
            self.mpc.set_objective(lterm=self.model.x['x']**2, mterm=self.model.u['u']**2)


if __name__ == '__main__':
    unittest.main()
