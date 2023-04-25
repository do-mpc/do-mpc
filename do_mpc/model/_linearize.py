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
import pdb
from ._linearmodel import LinearModel
from . import Model

def linearize(model:Model, 
              xss:np.ndarray = None, 
              uss:np.ndarray=None, 
              tvp0:np.ndarray = None, 
              p0:np.ndarray = None
              )->LinearModel:
    """Linearize the non-linear :py:class:`Model` to obtain a :py:class:`LinearModel` .
    The linearized model is required, e.g. for the :py:class:`do_mpc.controller.LQR` controller.
    
    This method uses the taylor expansion series to linearize non-linear model to linear model at the specified 
    set points. Linearized model retains the same variable names for all states, inputs with respect to the original model. 
    The non-linear model equation this method can solve is as follows:
    
    .. math::
        \\dot{x} = f(x,u)
            
    The above model is linearized around steady state set point :math:`x_{ss}` and steady state input :math:`u_{ss}`
    
    .. math::
        \\frac{\\partial f}{\\partial x}|_{x_{ss}} = 0 \\\\
        \\frac{\\partial f}{\\partial u}|_{u_{ss}} = 0
            
    The linearized model is as follows:
        
    .. math::
        \\Delta\\dot{x} = A \\Delta x + B \\Delta u
            
    Similarly, it can be extended to discrete time systems. Since the linearized model has only rate of change input and state. The names are appended with 'del' to differentiate 
    from the original model. This can be seen in the above model definition. Therefore, the solution of the lqr will be ``u`` and its corresponding ``x``. In order to fetch :math:`\\Delta u` 
    and :math:`\\Delta x`, setpoints has to be subtracted from the solution of lqr.
    
    Args:
        model : dynamic systems model 
        xss : Steady state state
        uss : Steady state input
        tvp0 : value for tvp variable
        p0 : value for parameter variable   
    
    Returns:
        Linearized Model    
    """
    #Check whether model setup is done
    assert model.flags['setup'] == True, 'Run this function after original model is setup'
    assert model.z.size == 0, 'Linearization around steady state is not supported for DAEs'    
    
    A,B,C,D = model.get_linear_system_matrices(xss,uss, tvp=tvp0, p=p0)
    
    # Check if A,B,C,D are constant or expressions
    all_constant = np.alltrue(
        [isinstance(A, np.ndarray), isinstance(B, np.ndarray), isinstance(C, np.ndarray), isinstance(D, np.ndarray)]
    )
    
    if all_constant:
        # If all are constant, linear model is initialized
        linearizedModel = LinearModel(model.model_type,model.symvar_type)
    else:
        # If not, LTV model is initialized
        raise NotImplementedError('LTV models are not yet implemented.')


    # Create new variables for linearized model
    model._transfer_variables(model, linearizedModel)
    n_x = model.n_x
    n_u = model.n_u
    
    # Check for trivial measurement equation
    if C.shape == (n_x,n_x) and (C == np.eye(n_x)).all():
        C = None
    if D.shape == (n_x,n_u) and (D == np.zeros((n_x,n_u))).all():
        D = None
    
    # Setup linearized model
    linearizedModel.setup(A,B,C,D)
    
    return linearizedModel