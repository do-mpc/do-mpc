
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

# %%
import sys 
import os
import numpy as np
import casadi.tools as castools

sys.path.append(os.path.join('..','..'))
sys.path.append(os.path.join('..','CSTR'))
import do_mpc
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator


# %%

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)

# %%
nlpdiff = do_mpc.differentiator.DoMPCDifferentiator(mpc)
nlpdiff.settings.check_rank = False
nlpdiff.settings.check_LICQ = False

# %%
simulator.x0 = np.array([0.5, 0.5, 134.14, 130.0]).reshape(-1,1)
# %%
mpc.make_step(simulator.x0)
# %%
dxdp, dlamdp = nlpdiff.differentiate()
# %%
dlamdp.shape
# %%
nlpdiff.status
# %%
nlpdiff.sens_num['dxdp', castools.indexf['_u', 0, 0], castools.indexf['_x0']]
# %%
