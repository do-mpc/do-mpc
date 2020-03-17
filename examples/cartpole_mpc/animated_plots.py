#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2020 Sergio Lucia, Alexandru Tatulea-Codrean
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
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, ImageMagickWriter
import pickle
plt.ion()

results  = do_mpc.data.load_results('./results/cartpole_results.pkl')

"""
Static plot Example
"""
static_graph = do_mpc.graphics.Graphics()

fig1, ax1 = plt.subplots(4, sharex=True, figsize=(8,6))
# Configure plot:
static_graph.add_line(var_type='_x', var_name='x', axis=ax1[0])
static_graph.add_line(var_type='_x', var_name='v', axis=ax1[1])
static_graph.add_line(var_type='_x', var_name='theta', axis=ax1[2])
static_graph.add_line(var_type='_u', var_name='F', axis=ax1[3])
ax1[0].set_ylabel('x [m]')
ax1[1].set_ylabel('v [m/sec]')
ax1[2].set_ylabel('$\Theta$ [rad]')
ax1[3].set_ylabel('Force [N]')
ax1[3].set_xlabel('Time [h]')

fig1.align_ylabels()


simu_lines = static_graph.plot_results(results['simulator'])

fig1.tight_layout()

plt.show()
#input('press any key.')


"""
Animation
"""
# This part animnates the time plots of states and inputs
dynamic_graph = do_mpc.graphics.Graphics()

fig2, ax2 = plt.subplots(4, sharex=True, figsize=(8,6))
fig2.suptitle("Inverted Pendulum NMPC: Animated Trends", fontsize=16)
# Configure plot:
dynamic_graph.add_line(var_type='_x', var_name='x', axis=ax2[0])
dynamic_graph.add_line(var_type='_x', var_name='v', axis=ax2[1])
dynamic_graph.add_line(var_type='_x', var_name='theta', axis=ax2[2])
dynamic_graph.add_line(var_type='_u', var_name='F', axis=ax2[3])
ax2[0].set_ylabel('x [m]')
ax2[1].set_ylabel('v [m/sec]')
ax2[2].set_ylabel('$\Theta$ [rad]')
ax2[3].set_ylabel('Force [N]')
ax2[3].set_xlabel('Time [sec]')

fig2.align_ylabels()

# And this part animates the pendulum sketch

fig3, ax3 = plt.subplots(1, sharex=True, figsize=(8,6))
fig3.suptitle("Inverted Pendulum NMPC: Animated CartPole", fontsize=16)
#fig3.tight_layout()

Theta_text = ax3.text(-1,-3, '', fontsize = 14)
F_text     = ax3.text(-1.2,-5, '', fontsize = 14)

cart       = patches.Rectangle((-1,-0.5),2.0, 1.0, edgecolor='none',facecolor='steelblue')
pendulum   = patches.Rectangle((-0.15,0),0.3, 4.0, edgecolor='none',facecolor='sienna')


output_format = 'gif' #'gif' #'mp4'

def update_trends(t_ind):
    dynamic_graph.reset_axes()
    simu_lines = dynamic_graph.plot_results(results['simulator'], t_ind=t_ind, linewidth=3)
    dynamic_graph.plot_predictions(results['mpc'], t_ind=t_ind, linewidth=1, linestyle='--')
    
def update_sketch(t_ind):
    global cart
    global pendulum
    
    
    X_k     = results['simulator']._x[t_ind,0]
    Theta_k = results['simulator']._x[t_ind,3]
    F_k     = results['simulator']._u[t_ind,0]
    
    cart.set_xy((X_k-1.5,-0.6))
    
    pendulum.set_xy((X_k-0.15,0.0))
    pendulum.angle = round(np.rad2deg(Theta_k),2)
    
    new_angle = "$\Theta=$" + str(round(np.rad2deg(Theta_k),2)) +"°"
    new_force = "$F_u=$"   + str(round(F_k,2)) +" N"
    Theta_text.set_text(new_angle)
    F_text.set_text(new_force)
    
    return cart, pendulum, Theta_text, F_text

def init_sketch():
    
    global cart
    global pendulum
    
    ax3.set_xlim(-10,10); ax3.set_ylim(-10,10)
    #ax3.set_xticks([])
    ax3.set_yticks([])
    #ax3.set_axis_off()
    ax3.set_xlabel("Cart position [m]")
    
    ax3.plot([-10,10],[0,0], color='sandybrown')  # Draw the middle line
    cart.set_width(3.0);     cart.set_height(1.2)
    pendulum.set_width(0.3); pendulum.set_height(4.0)
    
    ax3.add_patch(pendulum) # Draws the pendulum straight upright
    ax3.add_patch(cart)     # Draws the cart in the origin
  
    return cart, pendulum 

#anim_trends = FuncAnimation(fig2, update_trends, frames=results['simulator']._time.size, interval= 0.5, repeat=False)
anim_sketch = FuncAnimation(fig3, update_sketch, init_func=init_sketch, frames=results['simulator']._time.size, interval= 20, blit= True, repeat=False)

if 'mp4' in output_format:
    FFWriter = FFMpegWriter(fps=6, extra_args=['-vcodec', 'libx264'])
    anim_trends.save('anim.mp4', writer=FFWriter)
elif 'gif' in output_format:
    gif_writer = ImageMagickWriter(fps=3)
    anim_sketch.save('anim.gif', writer=gif_writer)
else:
    plt.show()
