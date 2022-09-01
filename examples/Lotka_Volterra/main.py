# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:22:33 2022

@author: jonak
"""

# Import files


# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

show_animation = True
store_results = False

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Initial state

x_0_0 = .5
x_1_0 = .7

x0 = np.array([x_0_0,x_1_0])

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

# Setup graphic
states_list = ['x_0', 'x_1']
input_list  = ['inp']
fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, states_list=states_list, inputs_list=input_list, figsize=(8,5))
plt.ion()

for k in range(25):
    u0 = mpc.make_step(x0)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)


    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)
input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'Lotka_Volterra_MINLP_MPC')



























