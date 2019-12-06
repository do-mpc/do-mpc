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

# Setup default graphic
fig, ax = configuration.setup_graphic()


for k in range(100):
    configuration.make_step_optimizer()
    configuration.make_step_simulator()
    configuration.make_step_estimator()
    #configuration.plot_animation()

configuration.plot_results()

configuration.save_results()
