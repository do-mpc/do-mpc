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
from matplotlib.animation import FuncAnimation, ImageMagickWriter


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
show_animation = True
store_results = False
create_gif = False
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
sampler.settings.closed_loop_flag = True
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

# Initialize graphic:
graphics = do_mpc.graphics.Graphics(simulator.data)


fig, ax = plt.subplots(5, sharex=True)
# Configure plot:
graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('T [K]')
ax[2].set_ylabel('$\Delta$ T [K]')
ax[3].set_ylabel('Q [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')
# Update properties for all prediction lines:
#for line_i in graphics.pred_lines.full:
#    line_i.set_linewidth(1)

#label_lines = graphics.result_lines['_x', 'C_a']+graphics.result_lines['_x', 'C_b']
#ax[0].legend(label_lines, ['C_a', 'C_b'])
#label_lines = graphics.result_lines['_x', 'T_R']+graphics.result_lines['_x', 'T_K']
#ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.align_ylabels()
fig.tight_layout()
plt.ion()


for k in range(50):
    u0 = approx_mpc.make_step(x0, clip_to_bounds=False)
    y_next = simulator.make_step(u0)
    x0 = estimator.make_step(y_next)

    if show_animation:
        graphics.plot_results(t_ind=k)
        #graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

# generate plot of profile
if create_gif:
    sim_graphics = do_mpc.graphics.Graphics(simulator.data)

    fig, ax = plt.subplots(5, sharex=True, figsize=(16,12))
    # Configure plot:
    sim_graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
    sim_graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
    sim_graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
    sim_graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
    sim_graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
    sim_graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
    sim_graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
    ax[0].set_ylabel('c [mol/l]')
    ax[1].set_ylabel('T [K]')
    ax[2].set_ylabel('$\Delta$ T [K]')
    ax[3].set_ylabel('Q [kW]')
    ax[4].set_ylabel('Flow [l/h]')
    ax[4].set_xlabel('time [h]')

    # Update properties for all prediction lines:
    #for line_i in sim_graphics.pred_lines.full:
    #    line_i.set_linewidth(2)
    ## Highlight nominal case:
    #for line_i in np.sum(sim_graphics.pred_lines['_x', :, :,0]):
    #    line_i.set_linewidth(5)
    #for line_i in np.sum(sim_graphics.pred_lines['_u', :, :,0]):
    #    line_i.set_linewidth(5)
    #for line_i in np.sum(sim_graphics.pred_lines['_aux', :, :,0]):
    #    line_i.set_linewidth(5)

    # Add labels
    label_lines = sim_graphics.result_lines['_x', 'C_a']+sim_graphics.result_lines['_x', 'C_b']
    ax[0].legend(label_lines, ['C_a', 'C_b'])
    label_lines = sim_graphics.result_lines['_x', 'T_R']+sim_graphics.result_lines['_x', 'T_K']
    ax[1].legend(label_lines, ['T_R', 'T_K'])

    fig.align_ylabels()

    def update(t_ind):
        print('Writing frame: {}.'.format(t_ind), end='\r')
        sim_graphics.plot_results(t_ind=t_ind)
        #sim_graphics.plot_predictions(t_ind=t_ind)
        sim_graphics.reset_axes()
        lines = sim_graphics.result_lines.full
        return lines

    n_steps = sim_graphics.data['_time'].shape[0]


    anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

    gif_writer = ImageMagickWriter(fps=5)
    anim.save('anim_appx.gif', writer=gif_writer)

# end
input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'CSTR_robust_MPC')
