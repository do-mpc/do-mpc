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
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    # Initialize model
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type)
    
    # Certain Parameters
    k1 = 25  # Constant rate [s^-1]
    k2 = 1   # Constant rate [s^-1]
    k3 = 1   # Constant rate [s^-1]
    
    # Initialize states, algebraic states and inputs
    Ca = model.set_variable('_x','Ca')
    Cb = model.set_variable('_x','Cb')
    Ad = model.set_variable('_x','Ad')
    
    Cain = model.set_variable('_u','Cain')
    
    Cc = model.set_variable('_z','Cc')
    
    # Differential algebraic equations
    model.set_rhs('Ca',-k1*Ca+Cain)
    model.set_rhs('Cb',k1*Ca-k2*Cb+k3*Cc)
    model.set_rhs('Ad',Cain)
    
    model.set_alg('exp',1+Ad-Ca-Cb-Cc)
    
    # setup the model
    model.setup()
    
    #DAE model converted to ODE
    daemodel = do_mpc.tools.dae2odeConversion.dae_to_ode_model(model)
    
    #converted ODE model is linearized
    linearmodel = daemodel.linearize()
    
    return model, daemodel, linearmodel