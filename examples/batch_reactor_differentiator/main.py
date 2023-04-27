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
import do_mpc.differentiator as differentiator
# from do_mpc.differentiator._nlpdifferentiator import get_do_mpc_nlp_sol, build_sens_sym_struct, assign_num_to_sens_struct

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator


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
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(8,5))
plt.ion()

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


# nlp_diff = differentiator.NLPDifferentiator.from_optimizer(mpc)
# nlp_diff = differentiator.NLPDifferentiator(mpc)
nlp_diff = differentiator.DoMPCDifferentiatior(mpc)


import cProfile
import pstats

# with cProfile.Profile() as pr:
pr = cProfile.Profile()
pr.enable()

    

for k in range(10):
    

    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)    
    x0 = estimator.make_step(y_next)


    track_nlp_obj.append(nlp_diff.nlp.copy())
    track_nlp_res.append(nlp_diff._get_do_mpc_nlp_sol(nlp_diff.optimizer).copy())

    tic = time.time()
    print("iteration: ", k)
    dx_dp_num, dlam_dp_num, residuals, LICQ_status, SC_status, where_cons_active = nlp_diff.differentiate()
    toc = time.time()
    print("Time to calculate sensitivities: ", toc-tic)
    assert LICQ_status==True
    assert residuals<=1e-12
    

    LICQ_status_list.append(LICQ_status)
    SC_status_list.append(SC_status)
    residuals_list.append(residuals)
    param_sens_list.append(dx_dp_num)

    sens_num = nlp_diff.get_dxdp_symstruct(dx_dp_num)
    du0dx0_num = sens_num["dxdp", indexf["_u",0,0], indexf["_x0"]]
    du0du_prev_num = sens_num["dxdp", indexf["_u",0,0], indexf["_u_prev"]]
  

    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

# input('Press any key to exit.')

pr.disable()

stats = pstats.Stats(pr)
# stats.sort_stats(pstats.SortKey.TIME)
stats.sort_stats("tottime")
stats.print_stats()
# dump stats to readable file
# stats.dump_stats("profile_stats_Sens.prof")