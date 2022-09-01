# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:27:24 2022

@author: jonak
"""

# Import updated do_mpc files

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

def template_model(symvar_type='SX'):
    model_type = 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)
    
    # Certain parameters
    c0 = .4
    c1 = .2
    
    # State struct (hunter and prey, optimization variables)
    x_0 = model.set_variable('_x', 'x_0')
    x_1 = model.set_variable('_x', 'x_1')
    
    # Input struct (optimization paramters)
    inp = model.set_variable('_u', 'inp', input_type_integer=True)
    
    # Differential equations
    model.set_rhs('x_0', x_0 - x_0*x_1 - c0*x_0*inp)
    model.set_rhs('x_1', -x_1 + x_0*x_1 - c1*x_1*inp)
    
    model.setup()
    
    return model