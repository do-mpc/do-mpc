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
    model_type = 'continuous' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Same example as shown in the Jupyter Notebooks.

    # Model variables:
    x = model.set_variable(var_type='_x', var_name='x')
    y = model.set_variable(var_type='_x', var_name='y')
    v = model.set_variable(var_type='_x', var_name='v')
    psi = model.set_variable(var_type='_x', var_name='psi')
    cog = model.set_variable('_x', 'cog')

    a=model.set_variable(var_type='_u', var_name='a')
    delta_f = model.set_variable(var_type='_u', var_name='delta_f')

    # Set point for the central mass:
    y_set = model.set_variable(var_type='_tvp', var_name='y_set')
    x_set = model.set_variable(var_type='_tvp', var_name='x_set')

    # Parameter for the MHE: Weighting of the arrival cost (parameters):
    P_p = model.set_variable(var_type='_p', var_name='P_p')

    # Time-varying parameter for the MHE: Weighting of the measurements (tvp):
    P_v = model.set_variable(var_type='_tvp', var_name='P_v', shape=(4, 4))

    # State measurements
    x_meas = model.set_meas('x_meas', x,meas_noise=True)
    y_meas = model.set_meas('y_meas', y,meas_noise=True)
    #v_meas = model.set_meas('v_meas', v)
    #psi_meas=model.set_meas('psi_meas',psi)
    #cog_meas = model.set_meas('cog_meas', cog)

    # Input measurements
    a_meas = model.set_meas('a_meas', a,meas_noise=True)
    delta_f_meas=model.set_meas('delta_f_meas',delta_f,meas_noise=True)
    l=3

    l_r=(1-cog)*l
    l_f=(cog)*l
    #l_r = model.set_variable('parameter', 'l_r')
    beta=arctan(l_r*tan(delta_f)/(l_r+l_f))


    model.set_rhs('x',v * cos(psi+beta),process_noise=True)
    model.set_rhs('y', v * sin(psi + beta),process_noise=True)
    model.set_rhs('v', a,process_noise=True)
    model.set_rhs('psi', v/l_r*sin(beta),process_noise=True)
    model.set_rhs('cog',DM(0),process_noise=True)



    model.setup()

    return model