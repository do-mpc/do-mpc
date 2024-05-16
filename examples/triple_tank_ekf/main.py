# imports
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
#from ... import do_mpc

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
    x0 = ekf.make_step(y_next = y_next, u_next = u0)

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


def visualize(x_data, x_hat_data):
    fig, ax = plt.subplots(model.n_x)
    fig.suptitle('EKF Observer')

    for i in range(model.n_x):
        ax[i].plot(x_data[i, :], label='real state')
        ax[i].plot(x_hat_data[i, :],"r--", label='estimated state')
        ax[i].set_xticklabels([])

    ax[-1].set_xlabel('time_steps')
    fig.legend()
    plt.show()

#print()
#print('Shapes:')
#print(x_data.shape)
#print(x_hat_data.shape)
x_data = np.concatenate(x_data, axis=1)
x_hat_data = np.concatenate(x_hat_data, axis=1)
if default_plot is False:
    visualize(x_data, x_hat_data)


input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([ekf, simulator], 'rot_oscillating_masses')
print('Individual Simulator data:')
print('Type:', type(simulator.data._x[-1]))
print('Shape:', simulator.data._x[-1].shape)

print('Individual EKF data:')
print('Type:', type(ekf.data._x[-1]))
print('Shape:', ekf.data._x[-1].shape)

print('Simulator data:')
print('Type:', type(simulator.data._x))
print('Shape:', simulator.data._x.shape)

print('EKF data:')
print('Type:', type(ekf.data._x))
print('Shape:', ekf.data._x.shape)