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


class TestModelSetVariable(unittest.TestCase):
    """Long and short names for ``var_type`` in :py:meth:`do_mpc.model.Model.set_variable` (see #471)."""

    NAMES = [
        ('states', '_x'),
        ('inputs', '_u'),
        ('algebraic', '_z'),
        ('parameter', '_p'),
        ('timevarying_parameter', '_tvp'),
    ]

    def test_long_and_short_names_are_equivalent(self):
        for long_name, short_name in self.NAMES:
            for var_type in (long_name, short_name):
                with self.subTest(var_type=var_type):
                    model = do_mpc.model.Model('continuous')
                    model.set_variable(var_type, 'var', shape=(2, 1))
                    self.assertIn('var', getattr(model, short_name)['name'])

    def test_integer_input_with_long_name(self):
        model = do_mpc.model.Model('continuous')
        model.set_variable('inputs', 'u_int', input_type_integer=True)
        self.assertIn('u_int', model._u['name'])

    def test_integer_flag_rejected_for_non_inputs(self):
        model = do_mpc.model.Model('continuous')
        with self.assertRaises(AssertionError):
            model.set_variable('states', 'x_int', input_type_integer=True)

    def test_invalid_var_type_rejected(self):
        model = do_mpc.model.Model('continuous')
        with self.assertRaises(AssertionError):
            model.set_variable('not_a_var_type', 'var')


if __name__ == '__main__':
    unittest.main()
