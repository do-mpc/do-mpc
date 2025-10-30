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
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc

# local imports
from template_model import template_model
from template_ekf import template_ekf
from template_simulator import template_simulator

model = template_model()
simulator = template_simulator(model)
simulator_ekf = template_simulator(model)
ekf = template_ekf(model)

""" User settings: """
show_animation = True
store_results = False

# simulation horizon
N_sim = 200
np.random.seed(1)
# setting up model variances with a generic value
q = 1e-3 * np.ones(model.n_x)
r = 1e-2 * np.ones(model.n_y)
#q = 0 * np.ones(model.n_x)
#r = 0 * np.ones(model.n_y)
Q = np.diag(q.flatten())
R = np.diag(r.flatten())

# initial states of model
x0_true = np.array([2, 2.8, 2.7]).reshape([-1, 1])
#x0 = np.zeros(model.n_x)
x0 = np.array([1.2, 1.4, 1.8]).reshape([-1, 1])

simulator.x0 = x0_true
ekf.x0 = x0
simulator_ekf.x0 = x0
simulator.set_initial_guess()
simulator_ekf.set_initial_guess()

ekf.set_initial_guess()

default_plot = False
alternate_default_plot = True

if default_plot:
    if alternate_default_plot:
        fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data)
        plt.ion()
    else:
        color = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(3,1, sharex=True, figsize=(10, 9))

        sim_plot = do_mpc.graphics.Graphics(simulator.data)
        ekf_plot = do_mpc.graphics.Graphics(ekf.data)

        ax[0].set_title('Estimated height of tank 1:')
        sim_plot.add_line('_x', 'x1', ax[0])
        ekf_plot.add_line('_x', 'x1', ax[0])

        ax[1].set_title('Estimated height of tank 2:')
        sim_plot.add_line('_x', 'x2', ax[1])
        ekf_plot.add_line('_x', 'x2', ax[1])

        ax[2].set_title('Estimated height of tank 3:')
        sim_plot.add_line('_x', 'x3', ax[2])
        ekf_plot.add_line('_x', 'x3', ax[2])


        ax[0].set_ylabel('tank \n height [m]')
        ax[1].set_ylabel('tank \n height [m]')
        ax[2].set_ylabel('tank \n height [m]')
        ax[2].set_xlabel('time [s]')

        for ax_i in ax:
            ax_i.axvline(1.0)

        fig.tight_layout()
        plt.ion()

# extra
x_data = [x0_true]
x_hat_data = [x0]

for k in range(N_sim):
    u0 = np.array([0.0001, 0.0001]).reshape([-1, 1])
    # Simulate with process and measurement noise

    #y_next = simulator.make_step(u0, v0=np.random.normal(0, np.sqrt(r)).reshape(-1, 1), w0=np.random.normal(0, np.sqrt(q)).reshape(-1, 1))
    #y_next = simulator.make_step(u0, v0=0.001*np.random.randn(model.n_v,1), w0 = 0.001*np.random.randn(model.n_v,1))
    y_next = simulator.make_step(u0, v0=0.001*np.random.randn(model.n_v,1))
    #print('shape of y_next:', y_next.shape)
    
    #x0 = mhe.make_step(y_next)
    x0 = ekf.make_step(y_next = y_next, u_next = u0, Q_k=Q, R_k=R)

    x_data.append(simulator.data._x[-1].reshape((-1,1)))
    x_hat_data.append(x0)


    if show_animation & default_plot is True & alternate_default_plot is False:
        #mpc_plot.plot_results()
        #mpc_plot.plot_predictions()
        ekf_plot.plot_results()
        sim_plot.plot_results()

        #mpc_plot.reset_axes()
        ekf_plot.reset_axes()
        sim_plot.reset_axes()
        plt.show()
        plt.pause(0.01)



# user settings
show_animation = True
store_results = False



# setting up the model
model = template_model()

# setting up a simulator, given the model
simulator = template_simulator(model)

# setting up Extended Kalman Filter (EKF), given the model
ekf = template_ekf(model)

# setting up model variances with a generic value
q = 1e-3 * np.ones(model.n_x)
r = 1e-2 * np.ones(model.n_y)
Q = np.diag(q.flatten())
R = np.diag(r.flatten())

# Initial covariance matrix of the EKF (if not set, it is initialized to the identity matrix)
ekf.P0 = np.eye(model.n_x)

# initial states of the simulator which is the real state of the system
x0_true = np.array([2, 2.8, 2.7]).reshape([-1, 1])

# and the initial state of the EKF which is a guess of the initial state of the system
x0 = np.array([1.2, 1.4, 1.8]).reshape([-1, 1])

# pushing initial condition to ekf and the simulator
simulator.x0 = x0_true
ekf.x0 = x0

# setting up initial guesses
simulator.set_initial_guess()
ekf.set_initial_guess()

# plot initialization
x_data = [x0_true]
x_hat_data = [x0]

# fix numpy random seed for reproducibility
np.random.seed(42)


# simulation of the plant
for k in range(200):

    # a step input is applied to the system
    u0 = np.array([0.0001, 0.0001]).reshape([-1, 1])

    # the simulator makes a step and returns the next state of the system
    y_next = simulator.make_step(u0, v0=0.001*np.random.randn(model.n_v,1))

    # the EKF makes a step and returns the next state of the system
    x0 = ekf.make_step(y_next = y_next, u_next = u0, Q_k=Q, R_k=R)

    # data stored for plots
    x_data.append(simulator.data._x[-1].reshape((-1,1)))
    x_hat_data.append(x0)

# a plotting function is defined to visualize the results
def visualize(x_data, x_hat_data):
    fig, ax = plt.subplots(model.n_x)
    fig.suptitle('EKF Observer')

    for i in range(model.n_x):
        ax[i].plot(x_data[i, :], label='real state')
        ax[i].plot(x_hat_data[i, :],"r--", label='estimated state')
        ax[i].set_xticklabels([])
        ax[i].set_ylabel('x' + str(i+1))

    ax[-1].set_xlabel('time_steps')
    ax[0].legend()
    plt.show()

# data converted to numpy arrays for plotting
x_data = np.concatenate(x_data, axis=1)
x_hat_data = np.concatenate(x_hat_data, axis=1)
visualize(x_data, x_hat_data)

# end
input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results(save_list=[ekf, simulator], result_name='results_triple_tank_ekf', result_path='results/', overwrite=True)
    
