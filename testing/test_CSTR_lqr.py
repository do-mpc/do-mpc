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

class TestCSTRLQR(unittest.TestCase):
    def setUp(self):
        default_path = copy.deepcopy(sys.path)
        
        sys.path.append('../examples/lqr_examples/CSTR_lqr/')
        import template_model
        import template_lqr
        import template_simulator
        
        self.template_model = reload(template_model)
        self.template_lqr = reload(template_lqr)
        self.template_simulator = reload(template_simulator)
        
        sys.path = default_path
        
    def test_SX(self):
        print('Testing SX implementation')
        self.CSTR_lqr('SX')
        
    def test_MX(self):
        print('Testing MX implementation')
        self.CSTR_lqr('MX')
        
    def CSTR_lqr(self, symvar_type):
        """
        Get configured do-mpc modules:
        """
        if symvar_type == 'SX':
            model,linearmodel = self.template_model.template_model(symvar_type)
            lqr = self.template_lqr.template_lqr(linearmodel)
            simulator = self.template_simulator.template_simulator(model)
        
            """
            Set initial state
            """
            # Set the initial state of simulator:
            C_a0 = 0
            C_b0 = 0
            T_R0 = 387.05
            T_J0 = 387.05

            x0 = np.array([C_a0, C_b0, T_R0, T_J0]).reshape(-1,1)
            simulator.x0 = x0
            # Steady state values
            F_ss = 0.002365    # [m^3/min]
            Q_ss = 18.5583     # [kJ/min]
    
            C_ass = 1.6329     # [kmol/m^3]
            C_bss = 1.1101     # [kmolm^3]
            T_Rss = 398.6581   # [K]
            T_Jss = 397.3736   # [K]
    
            uss = np.array([[F_ss],[Q_ss]])
            xss = np.array([[C_ass],[C_bss],[T_Rss],[T_Jss]])
            lqr.set_setpoint(xss=xss,uss=uss)
        
            """
            Run some steps:
            """
            for k in range(200):
                u0 = lqr.make_step(x0)
                y_next = simulator.make_step(u0)
                x0 = y_next
            
            """
            Compare results to reference run:
            """
            ref = do_mpc.data.load_results('./results/results_CSTR_LQR.pkl')

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
        
        else:
            self.assertRaises(ValueError, self.template_model.template_model, symvar_type)
if __name__ == '__main__':
    unittest.main()