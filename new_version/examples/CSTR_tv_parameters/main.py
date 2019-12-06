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
import pickle

from template_model import template_model
from template_optimizer import template_optimizer
from template_simulator import template_simulator
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

model = template_model()
optimizer = template_optimizer(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.state_feedback(model)

# Two alternatives to create a configuration and set the intial state:
# 1:
configuration = do_mpc.configuration(simulator, optimizer, estimator)
configuration.set_initial_state(x0, reset_history=True)
# 2:
configuration = do_mpc.configuration(simulator, optimizer, estimator, x0=x0)
# The default variant is to use the initial states that were independently defined for
# simulator, estimator and optimizer.


fig, ax = plt.subplots(2, sharex=True)
configuration.graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
configuration.graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
configuration.graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[1])
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('Q_heat [kW]')
plt.ion()

for k in range(100):
    configuration.make_step_optimizer()
    configuration.make_step_simulator()
    configuration.make_step_estimator()

    configuration.graphics.reset_axes()
    configuration.graphics.plot_results(optimizer.data)
    configuration.graphics.plot_predictions(optimizer.data, optimizer.opt_x_num)
    plt.show()
    input('next step')




opti_lines = configuration.graphics.plot_results(optimizer.data)
simu_lines = configuration.graphics.plot_results(simulator.data)

plt.sca(ax[0])
ax[0].add_artist(plt.legend(opti_lines[:2], ['Ca', 'Cb'], title='optimizer', loc=1))
plt.sca(ax[0])
ax[0].add_artist(plt.legend(simu_lines[:2], ['Ca', 'Cb'], title='Simulator', loc=2))
plt.show()
# with open('test.pkl', 'wb') as f:
#     pickle.dump(optimizer.data, f)
