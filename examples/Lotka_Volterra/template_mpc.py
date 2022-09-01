# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:28:51 2022

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
def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': 25,
        'n_robust': 0,
        'open_loop': 0,
        't_step': .3,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 2,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }
    
    mpc.set_param(**setup_mpc)
    
    x_0 = model.x['x_0']
    x_1 = model.x['x_1']
    u   = model.u['inp'] 
    
    mterm = (x_0-1)**2 + (x_1-1)**2 
    lterm = mterm
    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    # Optional
    mpc.set_rterm(inp=1)
    
    # Bounds 
    mpc.bounds['lower', '_x', 'x_0'] = 0.0
    mpc.bounds['lower', '_x', 'x_1'] = 0.0
    mpc.bounds['upper', '_x', 'x_0'] = 2.0
    mpc.bounds['upper', '_x', 'x_0'] = 2.0
    mpc.bounds['lower', '_u', 'inp'] = 0.0
    mpc.bounds['upper', '_u', 'inp'] = 1.0
    
    mpc.setup()
    
    return mpc





























    