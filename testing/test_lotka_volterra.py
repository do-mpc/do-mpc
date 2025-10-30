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
import pickle

from importlib import reload
import copy

do_mpc_path = '../'
if not do_mpc_path in sys.path:
    sys.path.append('../')

import do_mpc




class Testlotka_volterra(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """
        default_path = copy.deepcopy(sys.path)
        sys.path.append('../examples/lotka_volterra/')
        import template_model
        import template_mpc
        import template_simulator

        self.template_model = reload(template_model)
        self.template_mpc = reload(template_mpc)
        self.template_simulator = reload(template_simulator)
        sys.path = default_path

    def test_SX(self):
        print('Testing SX implementation')
        model = self.template_model.template_model('SX')
        self.lotka_volterra(model)

    def test_MX(self):
        print('Testing MX implementation')
        model = self.template_model.template_model('MX')
        self.lotka_volterra(model)

    def test_pickle_unpickle(self):
        print('Testing SX implementation with pickle / unpickle')
        # Test if pickling / unpickling works for the SX model:
        model = self.template_model.template_model('SX')
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Load the casadi structure
        with open('model.pkl', 'rb') as f:
            model_unpickled = pickle.load(f)
        self.lotka_volterra(model_unpickled)

    def lotka_volterra(self, model):
        """
        Get configured do-mpc modules:
        """

        mpc = self.template_mpc.template_mpc(model)
        simulator = self.template_simulator.template_simulator(model)
        estimator = do_mpc.estimator.StateFeedback(model)

        """
        Set initial state
        """

        # Initial state
        x_0_0 = .5
        x_1_0 = .7
        x0 = np.array([x_0_0,x_1_0])

        # pushing initial condition to mpc and the simulator
        mpc.x0 = x0
        simulator.x0 = x0

        # setting up initial guesses
        mpc.set_initial_guess()

        """
        Run some steps:
        """

        for k in range(2):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)

        """
        Store results (from reference run):
        """
        # do_mpc.data.save_results([mpc, simulator, estimator], 'results_lotka_volterra', overwrite=True)

        """
        Compare results to reference run:
        """
        ref = do_mpc.data.load_results('./results/results_lotka_volterra.pkl')

        test = ['_x', '_u', '_time', '_z']

        msg = 'Check if variable {var} for {module} is identical to previous runs: {check}. Max diff is {max_diff:.4E}.'
        for test_i in test:
            # Check MPC
            max_diff = np.max(np.abs(mpc.data.__dict__[test_i] - ref['mpc'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='MPC', check=check, max_diff=max_diff))

            # Check Simulator
            max_diff = np.max(np.abs(simulator.data.__dict__[test_i] - ref['simulator'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='Simulator', check=check, max_diff=max_diff))

            # Estimator
            max_diff = np.max(np.abs(estimator.data.__dict__[test_i] - ref['estimator'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='Estimator', check=check, max_diff=max_diff))



        # Store for test reasons
        try:
            do_mpc.data.save_results([mpc, simulator], 'test_save', overwrite=True)
        except:
            raise Exception()




if __name__ == '__main__':
    unittest.main()
