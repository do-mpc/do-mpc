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
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import pickle
plt.ion()

results  = do_mpc.data.load_results('./results/cartpole_mpc.pkl')

"""
Static plot Example
"""
graphics = do_mpc.graphics.Graphics()

fig, ax = plt.subplots(5, sharex=True, figsize=(8,6))
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
ax[2].set_ylabel('$\Delta T$ [K]')
ax[3].set_ylabel('$Q_{heat}$ [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')

fig.align_ylabels()


simu_lines = graphics.plot_results(results['simulator'])

ax[0].legend(simu_lines[:2], ['$C_a$', '$C_b$'], loc=1)
ax[1].legend(simu_lines[2:4], ['$T_R$', '$T_K$'], loc=1)

fig.tight_layout()

plt.show()
input('press any key.')


"""
Animation
"""
graphics = do_mpc.graphics.Graphics()

fig, ax = plt.subplots(5, sharex=True, figsize=(8,6))
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
ax[2].set_ylabel('$\Delta T$ [K]')
ax[3].set_ylabel('$Q_{heat}$ [kW]')
ax[4].set_ylabel('Flow [l/h]')
ax[4].set_xlabel('time [h]')

fig.align_ylabels()


output_format = 'gif'

def update(t_ind):
    graphics.reset_axes()
    simu_lines = graphics.plot_results(results['simulator'], t_ind=t_ind, linewidth=3)
    graphics.plot_predictions(results['mpc'], t_ind=t_ind, linewidth=1, linestyle='--')
    if t_ind==0:
        ax[0].legend(simu_lines[:2], ['$C_a$', '$C_b$'], loc=1)
        ax[1].legend(simu_lines[2:4], ['$T_R$', '$T_K$'], loc=1)


anim = FuncAnimation(fig, update, frames=99, repeat=False)

if 'mp4' in output_format:
    FFWriter = FFMpegWriter(fps=6, extra_args=['-vcodec', 'libx264'])
    anim.save('anim.mp4', writer=FFWriter)
elif 'gif' in output_format:
    gif_writer = ImageMagickWriter(fps=3)
    anim.save('anim.gif', writer=gif_writer)
else:
    plt.show()
