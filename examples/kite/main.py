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

# imports
import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import matplotlib.pyplot as plt

# local imports
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

# user settings
show_animation = True
store_results = False
store_animation = False

# setting up the model
model = template_model()

# setting up a mpc controller, given the model
w_ref = 6+10*np.random.rand()
E_0 = 5+3*np.random.rand()
h_min = 80+40*np.random.rand()
mpc = template_mpc(model, w_ref, E_0, h_min=h_min)

# setting up a simulator, given the model
simulator = template_simulator(model, w_ref, E_0)

# setting up an estimator, given the model
estimator = do_mpc.estimator.StateFeedback(model)

# Derive initial state from bounds:
lb_theta, ub_theta = mpc.bounds['lower','_x','theta'], mpc.bounds['upper','_x','theta']
lb_phi, ub_phi = mpc.bounds['lower','_x','phi'], mpc.bounds['upper','_x','phi']
lb_psi, ub_psi = mpc.bounds['lower','_x','psi'], mpc.bounds['upper','_x','psi']

# with mean and radius:
m_theta, r_theta = (ub_theta+lb_theta)/2, (ub_theta-lb_theta)/2
m_phi, r_phi = (ub_phi+lb_phi)/2, (ub_phi-lb_phi)/2
m_psi, r_psi = (ub_psi+lb_psi)/2, (ub_psi-lb_psi)/2

# How close can the intial state be to the bounds?
# tightness=1 -> Initial state could be on the bounds.
# tightness=0 -> Initial state will be at the center of the feasible range.
tightness = 0.6
theta_0 = m_theta-tightness*r_theta+2*tightness*r_theta*np.random.rand()
phi_0 = m_phi-tightness*r_phi+2*tightness*r_phi*np.random.rand()
psi_0 = m_psi-tightness*r_psi+2*tightness*r_psi*np.random.rand()

# Set the initial state of mpc, simulator and estimator:
x0 = np.array([theta_0, phi_0, psi_0]).reshape(-1,1)

# pushing initial condition to mpc, simulator and estimator
mpc.x0 = x0
simulator.x0 =x0
estimator.x0 = x0

# setting up initial guesses
mpc.set_initial_guess()

# Initialize graphic:
fig = plt.figure(figsize=(16,9))
plt.ion()

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

ax1 = plt.subplot2grid((4, 2), (0, 0), rowspan=4)
ax2 = plt.subplot2grid((4, 2), (0, 1))
ax3 = plt.subplot2grid((4, 2), (1, 1), sharex=ax2)
ax4 = plt.subplot2grid((4, 2), (2, 1))
ax5 = plt.subplot2grid((4, 2), (3, 1))

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
pred_lines = ax1.plot(phi_pred, theta_pred, color=color[0], linestyle='--', linewidth=1)

phi = mpc.data['_x', 'phi']
theta = mpc.data['_x', 'theta']
res_lines = ax1.plot(phi, theta, color=color[0])

# Height of kite
mpc_graphics.add_line(var_type='_aux', var_name='height_kite', axis=ax2)
mpc_graphics.add_line('_u','u_tilde',axis=ax3)
sim_graphics.add_line('_p','E_0', axis=ax4)
sim_graphics.add_line('_p','v_0', axis=ax5)

ax2.set_ylabel('kite height [m]')
ax3.set_ylabel('input [-]')
ax4.set_ylabel('E_0')
ax5.set_ylabel('v_0')

# simulation of the plant
n_steps = 200
for k in range(n_steps):

    # for the current state x0, mpc computes the optimal control action u0
    u0 = mpc.make_step(x0)

    # for the current state u0, computes the next state y_next
    y_next = simulator.make_step(u0)

    # for the current state y_next, estimates the next state x0
    x0 = estimator.make_step(y_next)

    # update the graphics
    if show_animation:
        phi_pred = mpc.data.prediction(('_x', 'phi'))[0]
        theta_pred = mpc.data.prediction(('_x', 'theta'))[0]
        for i in range(phi_pred.shape[1]):
            pred_lines[i].set_data(phi_pred[:,i], theta_pred[:,i])
        phi = mpc.data['_x', 'phi']
        theta = mpc.data['_x', 'theta']
        res_lines[0].set_data(phi, theta)
        ax1.relim()
        ax1.autoscale()

        mpc_graphics.plot_results()
        mpc_graphics.plot_predictions()
        mpc_graphics.reset_axes()
        sim_graphics.plot_results()
        sim_graphics.reset_axes()

        plt.show()
        plt.pause(0.01)

# Store animation:
if store_animation:
    from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
    def update(t_ind):
        phi_pred = mpc.data.prediction(('_x', 'phi'), t_ind)[0]
        theta_pred = mpc.data.prediction(('_x', 'theta'), t_ind)[0]
        for i in range(phi_pred.shape[1]):
            pred_lines[i].set_data(phi_pred[:,i], theta_pred[:,i])
        phi = mpc.data['_x', 'phi'][:t_ind]
        theta = mpc.data['_x', 'theta'][:t_ind]
        res_lines[0].set_data(phi, theta)
        ax1.relim()
        ax1.autoscale()

        mpc_graphics.plot_results(t_ind)
        mpc_graphics.plot_predictions(t_ind)
        mpc_graphics.reset_axes()
        sim_graphics.plot_results(t_ind)

    anim = FuncAnimation(fig, update, frames=n_steps, repeat=False)
    gif_writer = ImageMagickWriter(fps=20)
    anim.save('anim_kite.gif', writer=gif_writer)

input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([simulator], 'kite')
