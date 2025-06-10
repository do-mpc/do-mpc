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



class TestEKF(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """
        default_path = copy.deepcopy(sys.path)
        sys.path.append('../examples/triple_tank_ekf/')
        import template_model
        import template_ekf
        import template_simulator

        self.template_model = reload(template_model)
        self.template_ekf = reload(template_ekf)
        self.template_simulator = reload(template_simulator)
        sys.path = default_path

    def test_SX(self):
        print('Testing SX implementation')
        model = self.template_model.template_model('SX')
        self.TT_EKF(model)


    def test_pickle_unpickle(self):
        print('Testing SX implementation with pickle / unpickle')
        # Test if pickling / unpickling works for the SX model:
        model = self.template_model.template_model('SX')
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

        # Load the casadi structure
        with open('model.pkl', 'rb') as f:
            model_unpickled = pickle.load(f)
        self.TT_EKF(model_unpickled)

    def TT_EKF(self, model):
        """
        Get configured do-mpc modules:
        """


        # setting up a simulator, given the model
        simulator = self.template_simulator.template_simulator(model)

        # setting up Extended Kalman Filter (EKF), given the model
        ekf = self.template_ekf.template_ekf(model)

        # setting up model variances with a generic value
        q = 1e-3 * np.ones(model.n_x)
        r = 1e-2 * np.ones(model.n_y)
        Q = np.diag(q.flatten())
        R = np.diag(r.flatten())

        # initial states of the simulator which is the real state of the system
        x0_true = np.array([2, 2.8, 2.7]).reshape([-1, 1])

        # and the initial state of the EKF which is a guess of the initial state of the system
        x0 = np.array([1.2, 1.4, 1.8]).reshape([-1, 1])

        # pushing initial condition to ekf and the simulator
        simulator.x0 = x0_true
        ekf.x0 = x0

        # setting up initial guesses
        simulator.set_initial_guess()
        ekf.set_initial_guess()

        # fix numpy random seed for reproducibility
        np.random.seed(42)


        # simulation of the plant
        for k in range(200):

            # a step input is applied to the system
            u0 = np.array([0.0001, 0.0001]).reshape([-1, 1])

            # the simulator makes a step and returns the next state of the system
            y_next = simulator.make_step(u0, v0=0.001*np.random.randn(model.n_v,1))

            # the EKF makes a step and returns the next state of the system
            x0 = ekf.make_step(y_next = y_next, u_next = u0, Q_k=Q, R_k=R)




        """
        Compare results to reference run:
        """
        ref = do_mpc.data.load_results('./results/results_triple_tank_ekf.pkl')

        test = ['_x', '_u', '_time', '_z']

        msg = 'Check if variable {var} for {module} is identical to previous runs: {check}. Max diff is {max_diff:.4E}.'
        for test_i in test:

            # Check Simulator
            max_diff = np.max(np.abs(simulator.data.__dict__[test_i] - ref['simulator'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='Simulator', check=check, max_diff=max_diff))

            # Estimator
            max_diff = np.max(np.abs(ekf.data.__dict__[test_i] - ref['estimator'].__dict__[test_i]), initial=0)
            check = max_diff < 1e-8
            self.assertTrue(check, msg.format(var=test_i, module='Estimator', check=check, max_diff=max_diff))



        # Store for test reasons
        try:
            do_mpc.data.save_results([ekf, simulator], 'test_save', overwrite=True)
        except:
            raise Exception()




if __name__ == '__main__':
    unittest.main()
