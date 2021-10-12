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
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import unittest

from importlib import reload
import copy

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc

class TestIndustrialPoly(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """
        default_path = copy.deepcopy(sys.path)
        sys.path.append('../examples/industrial_poly/')
        import template_model
        import template_mpc
        import template_simulator

        self.template_model = reload(template_model)
        self.template_mpc = reload(template_mpc)
        self.template_simulator = reload(template_simulator)
        sys.path = default_path

    def test_SX(self):
        print('Testing SX implementation')
        self.industrialpoly('SX')

    def test_MX(self):
        print('Testing MX implementation')
        self.industrialpoly('MX')

    def industrialpoly(self, symvar_type):
        """
        Get configured do-mpc modules:
        """

        model = self.template_model.template_model(symvar_type)
        mpc = self.template_mpc.template_mpc(model)
        simulator = self.template_simulator.template_simulator(model)
        estimator = do_mpc.estimator.StateFeedback(model)

        """
        Set initial state
        """

        delH_R_real = 950.0
        c_pR = 5.0

        # x0 is a property of the simulator - we obtain it and set values.
        x0 = simulator.x0

        x0['m_W'] = 10000.0
        x0['m_A'] = 853.0
        x0['m_P'] = 26.5

        x0['T_R'] = 90.0 + 273.15
        x0['T_S'] = 90.0 + 273.15
        x0['Tout_M'] = 90.0 + 273.15
        x0['T_EK'] = 35.0 + 273.15
        x0['Tout_AWT'] = 35.0 + 273.15
        x0['accum_monom'] = 300.0
        x0['T_adiab'] = x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']

        # Finally, the controller gets the same initial state.
        mpc.x0 = x0

        # Which is used to set the initial guess:
        mpc.set_initial_guess()

        """
        Run some steps:
        """

        for k in range(5):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

        """
        Store results (from reference run):
        """
        #do_mpc.data.save_results([mpc, simulator, estimator], 'results_industrial_poly', overwrite=True)

        """
        Compare results to reference run:
        """
        ref = do_mpc.data.load_results('./results/results_industrial_poly.pkl')

        test = ['_x', '_u', '_time', '_z']

        for test_i in test:
            # Check MPC
            check = np.allclose(mpc.data.__dict__[test_i], ref['mpc'].__dict__[test_i])
            self.assertTrue(check)
            # Check Simulator
            check = np.allclose(simulator.data.__dict__[test_i], ref['simulator'].__dict__[test_i])
            self.assertTrue(check)
            # Estimator
            check = np.allclose(estimator.data.__dict__[test_i], ref['estimator'].__dict__[test_i])
            self.assertTrue(check)


        # Store for test reasons
        try:
            do_mpc.data.save_results([mpc, simulator], 'test_save', overwrite=True)
        except:
            raise Exception()

if __name__ == '__main__':
    unittest.main()
