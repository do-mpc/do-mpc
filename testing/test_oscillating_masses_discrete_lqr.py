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

class TestOscillatingMassesDiscreteLQR(unittest.TestCase):
    def setUp(self):
        default_path = copy.deepcopy(sys.path)
        
        sys.path.append('../examples/lqr_examples/oscillating_masses_discrete_lqr/')
        import template_model
        import template_lqr
        import template_simulator
        
        self.template_model = reload(template_model)
        self.template_lqr = reload(template_lqr)
        self.template_simulator = reload(template_simulator)
        
        sys.path = default_path
        
    def test_SX(self):
        print('Testing SX implementation')
        self.oscillating_masses_discrete_lqr('SX')
        
    def test_MX(self):
        print('Testing MX implementation')
        self.oscillating_masses_discrete_lqr('MX')
        
    def oscillating_masses_discrete_lqr(self, symvar_type):
        """
        Get configured do-mpc modules:
        """
        model = self.template_model.template_model(symvar_type)
        lqr = self.template_lqr.template_lqr(model)
        simulator = self.template_simulator.template_simulator(model)
        
        """
        Set initial state
        """
        x0 = np.array([[2],[1],[3],[1]])
        simulator.x0 = x0
        
        """
        Run some steps:
        """
        for k in range(50):
            u0 = lqr.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = y_next
            
        """
        Compare results to reference run:
        """
        ref = do_mpc.data.load_results('./results/results_oscillatingMasses_LQR.pkl')
        
        test = ['_x', '_u', '_time', '_z']
        
        for test_i in test:
            # Check Simulator
            check = np.allclose(simulator.data.__dict__[test_i], ref['simulator'].__dict__[test_i])
            self.assertTrue(check)
            
         # Store for test reasons
        try:
            do_mpc.data.save_results([simulator], 'test_save', overwrite=True)
        except:
            raise Exception()
            
if __name__ == '__main__':
    unittest.main()