# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:29:07 2022

@author: jonak
"""

# Import files

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_simulator: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)

    params_simulator = {
        'integration_tool': 'cvodes',
        'abstol': 1e-10,
        'reltol': 1e-10,
        't_step': .3
    }

    simulator.set_param(**params_simulator)

    p_num = simulator.get_p_template()


    def p_fun(t_now):
        return p_num

    simulator.set_p_fun(p_fun)

    simulator.setup()

    return simulator