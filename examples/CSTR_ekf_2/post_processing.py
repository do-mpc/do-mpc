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
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import pickle

plt.ion()

results  = do_mpc.data.load_results('./results/CSTR_robust_MPC.pkl')

"""
Static plot Example
"""
# Initialize graphic:
sim_graphics = do_mpc.graphics.Graphics(results['simulator'])


fig, ax = plt.subplots(5, sharex=True)
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


label_lines = sim_graphics.result_lines['_x', 'C_a']+sim_graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['C_a', 'C_b'])
label_lines = sim_graphics.result_lines['_x', 'T_R']+sim_graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.show()

input('press any key.')
"""
Animation
"""
# Initialize graphic:
mpc_graphics = do_mpc.graphics.Graphics(results['mpc'])

fig, ax = plt.subplots(5, sharex=True)
# Configure plot:
mpc_graphics.add_line(var_type='_x', var_name='C_a', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='C_b', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='T_R', axis=ax[1])
mpc_graphics.add_line(var_type='_x', var_name='T_K', axis=ax[1])
mpc_graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
mpc_graphics.add_line(var_type='_u', var_name='Q_dot', axis=ax[3])
mpc_graphics.add_line(var_type='_u', var_name='F', axis=ax[4])
ax[0].set_ylabel('c [mol/l]')
ax[1].set_ylabel('T [K]')
ax[2].set_ylabel('$\Delta$ T [K]')
ax[3].set_ylabel('Q [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')

# Update properties for all prediction lines:
for line_i in mpc_graphics.pred_lines.full:
    line_i.set_linewidth(1)
# Highlight nominal case:
for line_i in np.sum(mpc_graphics.pred_lines['_x', :, :,0]):
    line_i.set_linewidth(3)
for line_i in np.sum(mpc_graphics.pred_lines['_u', :, :,0]):
    line_i.set_linewidth(3)
for line_i in np.sum(mpc_graphics.pred_lines['_aux', :, :,0]):
    line_i.set_linewidth(3)

label_lines = mpc_graphics.result_lines['_x', 'C_a']+mpc_graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['C_a', 'C_b'])
label_lines = mpc_graphics.result_lines['_x', 'T_R']+mpc_graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.align_ylabels()
fig.tight_layout()

output_format = 'gif'

def update(t_ind):
    print('Writing frame: {}.'.format(t_ind))
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.plot_predictions(t_ind=t_ind)
    mpc_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines

n_steps = results['mpc']['_time'].shape[0]


anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

if 'mp4' in output_format:
    FFWriter = FFMpegWriter(fps=5, extra_args=['-vcodec', 'libx264'])
    anim.save('anim.mp4', writer=FFWriter)
elif 'gif' in output_format:
    gif_writer = ImageMagickWriter(fps=5, extra_args=['-MAGICK_MEMORY_LIMIT 4GiB'])
    anim.save('anim.gif', writer=gif_writer)
else:
    plt.show()
