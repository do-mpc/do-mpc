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
sys.path.append('../../')
import do_mpc
import scipy.io as sio
import matplotlib.pyplot as plt
plt.ion()


from template_model import template_model
from template_optimizer import template_optimizer
from template_simulator import template_simulator

# set initial value
c_pR = 5.0
m_W_0 = 10000.0
m_A_0 = 853.0*1.0  #3700.0
m_P_0 = 26.5
T_R_0  = 90 + 273.15
T_S_0  = 90 + 273.15
Tout_M_0  = 90 + 273.15
T_EK_0 = 35 + 273.15
Tout_AWT_0= 35 + 273.15

accum_momom_0   = 300.0

# This value is used here only to compute the initial condition of this state
# This should be changed in case the real value is different
delH_R_real = 950.0*1.00
T_adiab_0		= m_A_0*delH_R_real/((m_W_0+m_A_0+m_P_0)*c_pR)+T_R_0

x0   = np.array([m_W_0, m_A_0, m_P_0, T_R_0, T_S_0, Tout_M_0, T_EK_0, Tout_AWT_0, accum_momom_0,T_adiab_0]).reshape(-1,1)

model = template_model()
optimizer = template_optimizer(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.state_feedback(model)

optimizer._x0 = x0
simulator._x0 = x0
estimator._x0 = x0

configuration = do_mpc.configuration(simulator, optimizer, estimator)

for k in range(123):
    configuration.make_step_optimizer()
    configuration.make_step_simulator()
    configuration.make_step_estimator()

_x = simulator.data._x
_u = simulator.data._u
_t = simulator.data._time

for i in range(_x.shape[1]):
    plt.figure()
    plt.plot(_t, _x[:,i])

for i in range(_u.shape[1]):
    plt.figure()
    plt.plot(_t[0:-1], _u[:,i])
