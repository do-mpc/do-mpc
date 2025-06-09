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
from casadi.tools import *
import sys
import os
import torch
rel_do_mpc_path = os.path.join("..", "..")
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer
import matplotlib.pyplot as plt
import matplotlib
from do_mpc.approximateMPC import AMPCSampler
from do_mpc.approximateMPC import ApproxMPC, Trainer

# local imports
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator



# user settings
show_animation = True
store_results = False
create_gif = False
matplotlib.use("TkAgg")

# setting up the model
model = template_model()

# setting up a mpc controller, given the model
mpc = template_mpc(model, silence_solver=True)

# setting up a simulator, given the model
simulator = template_simulator(model)

# setting up an estimator, given the model
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8  # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5  # This is the controlled variable [mol/l]
T_R_0 = 134.14  # [C]
T_K_0 = 130.0  # [C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1, 1)
u0 = np.array([5.0, 0.0]).reshape(-1, 1)

# pushing initial condition to mpc and the simulator
mpc.u0=u0
mpc.x0 = x0
simulator.x0 = x0

# setting up initial guesses
mpc.set_initial_guess()
simulator.set_initial_guess() 

# approximate mpc initialization
approx_mpc = ApproxMPC(mpc)

# configuring approximate mpc settings
approx_mpc.settings.n_hidden_layers = 1
approx_mpc.settings.n_neurons = 50

# approximate mpc setup
approx_mpc.setup()


# initializing sampler for the approximate mpc
sampler = AMPCSampler(mpc)

dataset_name = 'my_dataset'
# configuring sampler settings
n_samples = 10000
sampler.settings.closed_loop_flag = True
sampler.settings.trajectory_length = 1
sampler.settings.n_samples = n_samples
sampler.settings.dataset_name = dataset_name

# sampler setup
sampler.setup()

# generating the samples
np.random.seed(42)  # for reproducibility
sampler.default_sampling()

# initializing trainer for the approximate mpc
trainer = Trainer(approx_mpc)

# configuring trainer settings
trainer.settings.dataset_name = dataset_name
trainer.settings.n_epochs = 3000
trainer.settings.show_fig =True
trainer.settings.save_fig = True
trainer.settings.save_history = True

# configuring scheduler settings
trainer.settings.scheduler_flag = True
trainer.scheduler_settings.cooldown = 0
trainer.scheduler_settings.patience = 50

# trainer setup
trainer.setup()


# training the approximate mpc with the sampled data
torch.manual_seed(42)  # for reproducibility
trainer.default_training()

# pushing initial condition to approx_mpc
approx_mpc.u0=u0

# Initialize graphic:
graphics = do_mpc.graphics.Graphics(simulator.data)
fig, ax = plt.subplots(5, sharex=True)

# Configure plot:
# adding each lines in th plot
graphics.add_line(var_type="_x", var_name="C_a", axis=ax[0])
graphics.add_line(var_type="_x", var_name="C_b", axis=ax[0])
graphics.add_line(var_type="_x", var_name="T_R", axis=ax[1])
graphics.add_line(var_type="_x", var_name="T_K", axis=ax[1])
graphics.add_line(var_type="_aux", var_name="T_dif", axis=ax[2])
graphics.add_line(var_type="_u", var_name="Q_dot", axis=ax[3])
graphics.add_line(var_type="_u", var_name="F", axis=ax[4])

# modifying the labels for the plot
ax[0].set_ylabel("c [mol/l]")
ax[1].set_ylabel("T [K]")
ax[2].set_ylabel("$\Delta$ T [K]")
ax[3].set_ylabel("Q [kW]")
ax[4].set_ylabel("Flow [l/h]")
ax[4].set_xlabel("time [h]")

# Update properties for all prediction lines:
for line_i in graphics.pred_lines.full:
    line_i.set_linewidth(1)

label_lines = graphics.result_lines["_x", "C_a"] + graphics.result_lines["_x", "C_b"]
ax[0].legend(label_lines, ["C_a", "C_b"])
label_lines = graphics.result_lines["_x", "T_R"] + graphics.result_lines["_x", "T_K"]
ax[1].legend(label_lines, ["T_R", "T_K"])

fig.align_ylabels()
fig.tight_layout()
plt.ion()

timer = Timer()

# simulation of the plant
sim_time=100
for k in range(sim_time):
    timer.tic()

    # for the current state x0, approx_mpc computes the optimal control action u0
    u0 = approx_mpc.make_step(x0,clip_to_bounds=True)
    timer.toc()

    # for the current state u0, computes the next state y_next
    y_next = simulator.make_step(u0)

    # for the current state y_next, computes the next state x0
    x0 = estimator.make_step(y_next)

    # update the graphics
    if show_animation:
        graphics.plot_results(t_ind=k)

        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

# print the results
timer.info()

input("Press any key to exit.")

# Store results:
if store_results:
    do_mpc.data.save_results([simulator], "CSTR_MPC")
