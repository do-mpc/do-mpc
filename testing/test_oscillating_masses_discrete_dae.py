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

sys.path.append('../')
import do_mpc
sys.path.pop(-1)

sys.path.append('../examples/oscillating_masses_discrete_dae/')
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
sys.path.pop(-1)


class TestOscillatingMassesDiscrete(unittest.TestCase):

    def test_oscillating_masses_discrete(self):
        """
        Get configured do-mpc modules:
        """

        model = template_model()
        mpc = template_mpc(model)
        simulator = template_simulator(model)
        estimator = do_mpc.estimator.StateFeedback(model)

        """
        Set initial state
        """

        np.random.seed(99)

        x0 = np.random.rand(model.n_x)-0.5
        mpc.x0 = x0
        simulator.x0 = x0
        estimator.x0 = x0

        # Use initial state to set the initial guess.
        mpc.set_initial_guess()
        # This is only meaningful for DAE systems.
        simulator.set_initial_guess()


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
        #do_mpc.data.save_results([mpc, simulator, estimator], 'results_oscillatingMasses_dae', overwrite=True)

        """
        Compare results to reference run:
        """
        ref = do_mpc.data.load_results('./results/results_oscillatingMasses_dae.pkl')

        test = ['_x', '_u', '_aux', '_time', '_z']

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




if __name__ == '__main__':
    unittest.main()
