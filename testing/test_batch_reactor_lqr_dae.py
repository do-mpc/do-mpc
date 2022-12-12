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

class TestBatchReactorLQRDAE(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """
        default_path = copy.deepcopy(sys.path)
        
        sys.path.append('../examples/lqr_examples/batch_reactor_lqr_dae/')
        import template_model
        import template_lqr
        import template_simulator
        
        self.template_model = reload(template_model)
        self.template_lqr = reload(template_lqr)
        self.template_simulator = reload(template_simulator)
        sys.path = default_path
        
    def test_SX(self):
        self.batch_reactor_lqr_dae('SX')

    def test_MX(self):
        self.batch_reactor_lqr_dae('MX')
        
    def batch_reactor_lqr_dae(self, symvar_type):
        """
        Get configured do-mpc modules:
        """
        t_sample = 0.5
        model,daemodel,linearmodel = self.template_model.template_model(symvar_type)
        model_dc = linearmodel.discretize(t_sample = 0.5)
        lqr = self.template_lqr.template_lqr(model_dc)
        simulator = self.template_simulator.template_simulator(linearmodel)
        
        """
        Set initial state
        """
        Ca0 = 1
        Cb0 = 0
        Ad0 = 0
        Cain0 = 0
        Cc0 = 0
        x0 = np.array([[Ca0],[Cb0],[Ad0],[Cain0],[Cc0]])
        
        simulator.x0 = x0
        
        """
        Set set points
        """
        Ca_ss = 0
        Cb_ss = 2
        Ad_ss = 3
        Cain_ss = 0
        Cc_ss = 2
        
        xss = np.array([[Ca_ss],[Cb_ss],[Ad_ss],[Cain_ss],[Cc_ss]])
        uss = model_dc.get_steady_state(xss = xss)
        lqr.set_setpoint(xss = xss, uss = uss)
        
        """
        Run MPC main loop:
        """
        for k in range(50):
            u0 = lqr.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = y_next
            
        """
        Compare results to reference run:
        """
        ref = do_mpc.data.load_results('./results/results_batch_reactor_LQR_DAE.pkl')

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