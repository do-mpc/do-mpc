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

from casadi.tools import *
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer
import matplotlib.pyplot as plt
import matplotlib


from do_mpc.approximateMPC import Sampler
from do_mpc.approximateMPC import ApproxMPC, Trainer

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

from matplotlib import rcParams

# plot setup
rcParams['axes.grid'] = True
rcParams['font.size'] = 18


""" User settings: """
show_animation = False
store_results = False
matplotlib.use('TkAgg')

model = template_model()
mpc = template_mpc(model,silence_solver=True)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

# pushing to class
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()
simulator.set_initial_guess()


# approximate mpc
approx_mpc = ApproxMPC(mpc)
approx_mpc.settings.n_hidden_layers = 3
approx_mpc.settings.n_neurons = 50
approx_mpc.setup()


# sampler
n_samples = 100
sampler = Sampler(mpc)
sampler.settings.closed_loop_flag = False
sampler.settings.trajectory_length = 5
sampler.settings.n_samples = n_samples
sampler.settings.data_dir = './sampled_data'
sampler.setup()
sampler.default_sampling()


# trainer
trainer = Trainer(approx_mpc)
trainer.settings.n_samples = n_samples
trainer.settings.n_epochs = 1000
trainer.settings.scheduler_flag = True
trainer.scheduler_settings.cooldown = 0
trainer.scheduler_settings.patience = 10
trainer.settings.show_fig =True
trainer.settings.save_fig = True
trainer.settings.save_history = True
trainer.settings.data_dir = './sampled_data'
trainer.setup()
trainer.default_training()


# saving data
approx_mpc.save_to_state_dict('approx_mpc.pth')
approx_mpc.load_from_state_dict('approx_mpc.pth')
# appx mpc end

for k in range(50):
    u0 = approx_mpc.make_step(x0, clip_to_bounds=False)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

# generate plot of profile
fig, ax, graphics = do_mpc.graphics.default_plot(simulator.data, figsize=(16,9))
graphics.plot_results()
graphics.reset_axes()
plt.show()

# end
input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'CSTR_robust_MPC')
