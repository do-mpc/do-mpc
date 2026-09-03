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
from casadi import DM

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc
from do_mpc._casadi_compat import castools


# Reference result written with casadi <= 3.7, i.e. a pickle that carries the
# old 'casadi.tools.structure3' module path.
LEGACY_PICKLE = './results/results_lotka_volterra.pkl'

# Symbols do-mpc reads off castools. Up to casadi 3.7 these leaked into
# casadi.tools via 'from casadi import *'; since 3.8 they live in casadi only.
CASADI_SYMBOLS = [
    'DM', 'MX', 'SX', 'Function', 'collocation_points', 'evalf', 'horzcat',
    'integrator', 'inv', 'jacobian', 'nlpsol', 'substitute', 'sum1', 'sum2',
    'tangent', 'vertcat',
]

# Symbols that are genuinely part of casadi.tools in every supported version.
CASADI_TOOLS_SYMBOLS = [
    'entry', 'indexf', 'struct_MX', 'struct_SX', 'struct_symMX', 'struct_symSX',
]


class TestCasadiCompat(unittest.TestCase):
    """Guards against the casadi 3.8 namespace changes.

    casadi 3.8.0 stopped re-exporting the casadi namespace from casadi.tools
    and renamed casadi.tools.structure3 to casadi.tools.structure.
    """

    def test_casadi_symbols_resolve(self):
        """Symbols that used to leak into casadi.tools are still reachable."""
        for name in CASADI_SYMBOLS:
            self.assertTrue(hasattr(castools, name),
                            'castools.{} does not resolve'.format(name))

    def test_casadi_tools_symbols_resolve(self):
        """Genuine casadi.tools symbols are still reachable."""
        for name in CASADI_TOOLS_SYMBOLS:
            self.assertTrue(hasattr(castools, name),
                            'castools.{} does not resolve'.format(name))

    def test_structure3_alias(self):
        """structure3 and structure refer to the same module on all versions."""
        self.assertIs(castools.structure3, castools.structure)
        self.assertTrue(hasattr(castools.structure3, 'DMStruct'))

    def test_structure3_importable(self):
        """The old module path stays importable, so old pickles keep working."""
        import casadi.tools.structure3
        self.assertIs(casadi.tools.structure3, castools.structure3)

    def test_load_legacy_results(self):
        """Result files written with casadi <= 3.7 still load.

        This is the regression that is invisible to the rest of the suite: the
        module was importable but every saved result raised ModuleNotFoundError.
        """
        # Make sure the fixture still is a pre-3.8 pickle, so that this test
        # cannot silently pass if the reference files are ever regenerated.
        with open(LEGACY_PICKLE, 'rb') as f:
            self.assertIn(b'casadi.tools.structure3', f.read(),
                          '{} is no longer a pre-casadi-3.8 pickle, the '
                          'regression it guards is no longer covered'
                          .format(LEGACY_PICKLE))

        results = do_mpc.data.load_results(LEGACY_PICKLE)
        self.assertIn('mpc', results)
        self.assertGreater(results['mpc']['_x'].shape[0], 0)


class TestAuxExpressionSparsity(unittest.TestCase):
    """Guards the casadi 3.8 structural zero change in the _aux struct.

    A Model that never calls set_expression carries a single aux entry, the
    default DM(0). casadi 3.8 propagates that structural zero through an MX
    function call, so _aux_expression_fun returns a 1x1 with nnz 0 while the
    struct slot it is written into is dense. struct.__setitem__ compares
    sparsity patterns and rejects it, which broke MPC and MHE setup for every
    such model. casadi 3.7 returned a dense result and never tripped this.
    """

    def _model(self, symvar_type):
        model = do_mpc.model.Model('continuous', symvar_type)
        x = model.set_variable('_x', 'x')
        u = model.set_variable('_u', 'u')
        model.set_rhs('x', -x + u)
        # Deliberately no set_expression: '_aux' holds only the default DM(0).
        model.setup()
        self.assertEqual(list(model['_aux'].keys()), ['default'])
        return model

    def test_mpc_setup_without_expressions(self):
        for symvar_type in ['SX', 'MX']:
            with self.subTest(symvar_type=symvar_type):
                mpc = do_mpc.controller.MPC(self._model(symvar_type))
                mpc.settings.supress_ipopt_output()
                mpc.set_param(n_horizon=2, t_step=0.1, store_full_solution=True)
                mpc.set_objective(mterm=DM(0), lterm=DM(0))
                mpc.set_rterm(u=1e-2)
                mpc.setup()

    def test_mhe_setup_without_expressions(self):
        for symvar_type in ['SX', 'MX']:
            with self.subTest(symvar_type=symvar_type):
                model = do_mpc.model.Model('continuous', symvar_type)
                x = model.set_variable('_x', 'x')
                u = model.set_variable('_u', 'u')
                model.set_rhs('x', -x + u, process_noise=True)
                model.set_meas('y', x, meas_noise=True)
                model.setup()
                self.assertEqual(list(model['_aux'].keys()), ['default'])

                mhe = do_mpc.estimator.MHE(model)
                mhe.settings.supress_ipopt_output()
                mhe.set_param(n_horizon=2, t_step=0.1, store_full_solution=True,
                              meas_from_data=True)
                mhe.set_default_objective(P_x=np.eye(1), P_v=np.eye(1),
                                          P_w=np.eye(1))
                mhe.setup()


if __name__ == '__main__':
    unittest.main()
