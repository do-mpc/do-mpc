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
from do_mpc.tools import save_pickle, load_pickle

class TestSamplingTools(unittest.TestCase):
    def setUp(self):
        """Add path of test case and import the modules.
        If this test isn't the first to run, the modules need to be reloaded.
        Reset path afterwards.
        """
        default_path = copy.deepcopy(sys.path)
        sys.path.append('../examples/tools/sampling/regular/test_fun/')
        import sampling_test

        self.test_function = sampling_test.main
        sys.path = default_path

    def test_sampling(self):
        res, res1, res2 = self.test_function()

        # Option to store new reference results:
        # save_pickle('./results/res_sampling_test_test_fun', {'res': res, 'res1': res1, 'res2': res2})

        ref = load_pickle('./results/res_sampling_test_test_fun.pkl')

        self.assertTrue(res == ref['res'])
        self.assertTrue(res1 == ref['res1'])
        self.assertTrue(res2 == ref['res2'])


if __name__ == '__main__':
    unittest.main()

