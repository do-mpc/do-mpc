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
from . import Model
import casadi.tools as castools
   
def dae2odeconversion(model:Model)->Model:
    """Converts index-1 DAE system to ODE system.
        
    This method utilizes the differentiation method of converting index-1 DAE systems to ODE systems. This method
    cannot handle higher index DAE systems. The DAE system is as follows:
            
    .. math::
        \\dot{x} = f(x,u,z) \\\\
            0 = g(x,u,z)
                
    where :math:`x` is the states, :math:`u` is the input and :math:`z` is the algebraic states of the system.
    Differentiation method is as follows:
    
    .. math::
        \\dot{z} = -\\frac{\\partial g}{\\partial z}^{-1}\\frac{\\partial g}{\\partial x}f-\\frac{\\partial g}{\\partial z}^{-1}\\frac{\\partial g}{\\partial u}\\dot{u}
            
    Therefore the converted ODE system looks like:

    .. math::
        \\begin{pmatrix} \\dot{x} \\\\ \\dot{u} \\\\ \\dot{z} \\end{pmatrix} = \\begin{pmatrix} f(x,u,z) \\\\ q \\\\ g(x,u,z) \\end{pmatrix}
            
    where :math:`\\dot{x},\\dot{u},\\dot{z}` are the states of the model and q is the input to the model. Similarly, it can be extended to discrete time systems.
    The dae to ode converted model assumes that converted algebraic states and states measurements are available.
    
    Args:
        model : Index-1 DAE model
        
    Returns:
        Converted ODE Model
    """
    #Check whether model setup is done
    assert model.flags['setup'] == True, 'Run this function after original model is setup'

    #Initializing new model
    daeModel = Model(model.model_type,model.symvar_type)
    
    #Setting states and inputs
    for key in range(np.size(model.x.keys())):
        daeModel.set_variable('_x',model.x.keys()[key],model.x[model.x.keys()[key]].size())
    for key in range(np.size(model.u.keys())-1):
        daeModel.set_variable('_x',model.u.keys()[key+1],model.u[model.u.keys()[key+1]].size())
    for key in range(np.size(model.z.keys())-1):
        daeModel.set_variable('_x',model.z.keys()[key+1],model.z[model.z.keys()[key+1]].size())
    for key in range(np.size(model.p.keys())-1):
        daeModel.set_variable('_p',model.p.keys()[key+1],model.p[model.p.keys()[key+1]].size())
    for key in range(np.size(model.tvp.keys())-1):
        daeModel.set_variable('_tvp',model.tvp.keys()[key+1],model.tvp[model.tvp.keys()[key+1]].size())
    q = daeModel.set_variable('_u','q',(model.n_u,1))
        
    #Extracting variables
    x_new = daeModel.x[model.x.keys()]
    u_new = daeModel.x[model.u.keys()[1:]]
    z_new = daeModel.x[model.z.keys()[1:]]
    tvp_new = daeModel.tvp[model.tvp.keys()[1:]]
    p_new = daeModel.p[model.p.keys()[1:]]
            
    #Converting rhs eq. with respect to variables of linear model of same name
    rhs = model._rhs_fun(castools.vertcat(*x_new),castools.vertcat(*u_new),castools.vertcat(*z_new),castools.vertcat(*tvp_new),castools.vertcat(*p_new),model.w)
    rhs_new = castools.substitute(rhs,model.w.cat,np.zeros(model.n_w).reshape(model.n_w,1))
    x_count = 0
    for i in range(np.size(model.x.keys())):
        if daeModel.x.keys()[i]+'_noise' in model.w.keys():
            daeModel.set_rhs(model.x.keys()[i],rhs_new[x_count:x_count+model.x[model.x.keys()[i]].size()[0]],process_noise=(True))
            x_count += model.x[model.x.keys()[i]].size()[0]
        else:
            daeModel.set_rhs(model.x.keys()[i],rhs_new[x_count:x_count+model.x[model.x.keys()[i]].size()[0]])
            x_count += model.x[model.x.keys()[i]].size()[0]
    alg = model._alg_fun(castools.vertcat(*x_new),castools.vertcat(*u_new),castools.vertcat(*z_new),castools.vertcat(*tvp_new),castools.vertcat(*p_new),daeModel.w)
    rhs_mod = castools.substitute(rhs,model.w.cat,daeModel.w.cat)
    z_next = -castools.inv(castools.jacobian(alg,castools.vertcat(*z_new)))@castools.jacobian(alg,castools.vertcat(*x_new))@rhs_mod-castools.inv(castools.jacobian(alg,castools.vertcat(*z_new)))@castools.jacobian(alg,castools.vertcat(*u_new))@q
    
    for j in range(np.size(model.u.keys())-1):
        daeModel.set_rhs(model.u.keys()[j+1],daeModel.u['q',j])
    
    z_count = 0
    for k in range(np.size(model.z.keys())-1):
        daeModel.set_rhs(model.z.keys()[k+1],z_next[z_count:z_count+model.z[model.z.keys()[k+1]].size()[0]])
        z_count += model.z[model.z.keys()[k+1]].size()[0]
    
    #setting up the model
    daeModel.setup()
    print('The states of the new model are {}' .format(daeModel.x.keys()))
    return daeModel
