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

from template_model import template_model
from template_optimizer import template_optimizer
from template_simulator import template_simulator

model = template_model()
optimizer = template_optimizer(model)
# simulator = template_simulator(model)
# estimator = do_mpc.estimator.state_feedback(model)

# configuration = do_mpc.configuration(simulator, optimizer, estimator)

opt_p_num = optimizer.opt_p_num
opt_p_num['_x0'] = optimizer._x0['x']

optimizer.solve()
pdb.set_trace()

# Example for storing data
optimizer.data.update(_x=optimizer.opt_x_num['_x', 0, 0, 0])
optimizer.data.update(_u=optimizer.opt_x_num['_u', 0, 0])
optimizer.data.update(_time=0)

X = horzcat(*optimizer.opt_x_num['_x', :, 0, 0])

plt.plot(X.T)
plt.show()
