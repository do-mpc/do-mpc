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
import copy
sys.path.append('../../')
import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import logging

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

logging.basicConfig( level=logging.INFO)


""" User settings: """
show_animation = False

"""
Get configured do-mpc modules:
"""

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

"""
Set initial state
"""

X_s_0 = 1.0 # This is the initial concentration inside the tank [mol/l]
S_s_0 = 0.5 # This is the controlled variable [mol/l]
P_s_0 = 0.0 #[C]
V_s_0 = 120.0 #[C]
x0 = np.array([X_s_0, S_s_0, P_s_0, V_s_0])


mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()


"""
Run MPC main loop:
"""

# run stats
LICQ_status_list = []
SC_status_list = []
residuals_list = []
param_sens_list = []
track_nlp_obj = []
track_nlp_res = []



import cProfile
import pstats

# with cProfile.Profile() as pr:
pr = cProfile.Profile()
pr.enable()


class ASMPC:
    def __init__(self, mpc):
        self.mpc = mpc
        self.nlp_diff = do_mpc.differentiator.DoMPCDifferentiatior(mpc)
        self.nlp_diff.settings.check_LICQ = False
        self.nlp_diff.settings.check_rank = False
        self.nlp_diff.settings.lin_solver = 'scipy'

        self._u_data = [mpc.u0.cat.full().reshape(-1,1)]

    def make_step(self, x0):
        x0 = x0.reshape(-1,1)

        self.nlp_diff.differentiate()


        x_prev = self.mpc.x0.cat.full().reshape(-1,1)
        u0 = self.mpc.u0.cat.full().reshape(-1,1)
        u_prev = self.mpc.opt_p_num['_u_prev'].full().reshape(-1,1)

        

        du0dx0_num = self.nlp_diff.sens_num["dxdp", indexf["_u",0,0], indexf["_x0"]]
        du0du_prev_num = self.nlp_diff.sens_num["dxdp", indexf["_u",0,0], indexf["_u_prev"]].full()

        A = np.eye(self.mpc.model.n_u)-du0du_prev_num

        u_next  = np.linalg.inv(A)@(u0 + du0dx0_num @ (x0 - x_prev) - du0du_prev_num @ (u0))
        # u_next = u0 + du0dx0_num @ (x0 - x_prev) - du0du_prev_num @ (u0 - u_prev)

        self._u_data.append(u_next)

        return u_next
    
    @property
    def u_data(self):
        return np.hstack(self._u_data)


asmpc = ASMPC(mpc)

for k in range(30):
    
    if k>0:
        u0_approx = asmpc.make_step(x0)
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)    
    x0 = estimator.make_step(y_next)

  


pr.disable()

stats = pstats.Stats(pr)

fig, ax = plt.subplots(1,1)

ax.plot(asmpc.u_data.T, '-x', label="approx")
ax.plot(mpc.data['_u'], '-x', label="mpc")
ax.legend()

plt.show(block=True)