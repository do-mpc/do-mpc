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

import tkinter as Tk
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.backends import _backend_tk
import matplotlib.patches as patches
import matplotlib.animation as animation
import pickle
import time

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
from opcmodules import RealtimeTrigger

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)

# Set the initial state of mpc and simulator:
x_0     = -2.0 # This is the initial position of the cart on the X-axis
v_0     = 0.0  # This is the initial velocity
theta_0 = 0.3  # This is the initial vertical orientation of the pendulum
omega_0 = 0.0  # And finally the initial angular velocity
x0 = np.array([x_0, v_0, theta_0, omega_0]).reshape(-1,1)


mpc.set_initial_state(x0, reset_history=True)
simulator.set_initial_state(x0, reset_history=True)

x00 = simulator._x0.cat.toarray()
u00 = mpc._u0.cat.toarray()
time_list = []
elapsed_time = 0
nmpc_iteration = 0
max_iterations = 100


"""            The GUI is created in this section
----------------------------------------------------------------------------
"""
def draw_pendulum_sketch(X_now, U_now, t_now):
  
    X_k     = X_now[0]
    Theta_k = X_now[2]
    F_k     = U_now[0]
    
    cart.set_xy((X_k-1.5,-0.6))
    
    pendulum.set_xy((X_k-0.15,0.0))
    pendulum.angle = np.rad2deg(Theta_k)
    
    new_angle = "$\Theta=$" + str(round(np.rad2deg(Theta_k),2)) +"°"
    new_force = "$F_u=$"   + str(round(F_k,2)) +" N"
    new_time  = "Elapsed: " + str(round(t_now,3)) + " sec"
    Theta_text.set_text(new_angle)
    Force_text.set_text(new_force)
    Time_text.set_text(new_time)
    
    return cart, pendulum, Theta_text, Force_text, Time_text

def update_canvas(x0,u0,t):
    # Update the sketch
    draw_pendulum_sketch(x0.reshape(-1),u0.reshape(-1),t)
    graph.draw()
    
    # Display the figure on canvas
    figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
    figure_w, figure_h = int(figure_w), int(figure_h)
    photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)
    canvas.create_image(320, 220 , image=photo)
    
    figure_canvas_agg = FigureCanvasAgg(fig)
    figure_canvas_agg.draw()
    
    _backend_tk.blit(photo, figure_canvas_agg.get_renderer()._renderer, (0, 1, 2, 3)) 
        
sg.theme('Reddit')	
# All the stuff inside your window.
layout = [  
    [sg.Canvas(size=(640,480), key='plot')],
    [sg.Text('1) do-mpc                  '), sg.Button('One NMPC Step',key='one_step'), sg.Button('Run NMPC steps:',key='run_nmpc'), sg.InputText(default_text='20',key='nr_iter', size=(10,20))],
    [sg.Text('2) Manual control:  F='), sg.Slider(range=(-10, 10), tick_interval=2, orientation='h', size=(24, 20), default_value=0, key='push_F'), sg.Button('Push',key='push'), sg.Button('Run', key='run_manually')],
    [sg.Text('3) Your new controller:', ), sg.InputText(default_text='your_controll_function'), sg.Button('Run',key='run_new')],
    [sg.Button('Stop'), sg.Button('Reset'), sg.Button('Close')] 
    ]

# Create the Window
window = sg.Window('CartPendulum Control', layout)
window.Finalize()  # needed to access the canvas element prior to reading the window

canvas_elem = window['plot']

fig = Figure()
ax = fig.add_subplot(111)
ax.set_xlabel("Cart position [m]")
ax.set_xlim(-10,10); ax.set_ylim(-10,10)
ax.set_yticks([])

Theta_text = ax.text(-2.0,-6, '', fontsize = 11)
Force_text = ax.text(-2.2,-8, '', fontsize = 11)
Time_text  = ax.text(-3.0,7.0, '', fontsize = 11)
cart     = patches.Rectangle((-3.5,-0.5),2.0, 1.0, edgecolor='none',facecolor='lightblue')
pendulum = patches.Rectangle((-2.15,0),0.3, 4.0, edgecolor='none',facecolor='sienna')
cart.set_width(3.0);     cart.set_height(1.2)
pendulum.set_width(0.3); pendulum.set_height(4.0)

ax.plot([-10,10],[0,0], color='black', linewidth = 0.5, linestyle='-.')  # Draw the middle line
ax.add_patch(cart)       # Place the cart on the sketch
ax.add_patch(pendulum)   # Place the pendulum on the sketch


graph = FigureCanvasTkAgg(fig, master=canvas_elem.TKCanvas)
canvas = canvas_elem.TKCanvas

graph.draw()
        
figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
figure_w, figure_h = int(figure_w), int(figure_h)
photo = Tk.PhotoImage(master=canvas, width=figure_w, height=figure_h)
canvas.create_image(320, 220, image=photo)

figure_canvas_agg = FigureCanvasAgg(fig)
figure_canvas_agg.draw()

_backend_tk.blit(photo, figure_canvas_agg.get_renderer()._renderer, (0, 1, 2, 3)) 

# 
"""   Event Loop to process "events" and get the "values" of the inputs
----------------------------------------------------------------------------
"""

while True:
    event, values = window.read()
    if event in (None, 'Close'):	# if user closes window or clicks close
        break
    print('Events ', event)
    print('Values ', values)
    
    if event == 'Reset':
        nmpc_iteration = 0
        mpc.set_initial_state(x0, reset_history=True)
        simulator.set_initial_state(x0, reset_history=True)
        time_list = []
        elapsed_time = 0.0
        # Reset the sketch to initial conditions
        update_canvas(x00, u00, 0.0)
        
    if event == 'push':
        # Simulate one step with the user provided push force
        u0 = np.array(values['push_F']).reshape(1,1)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
        elapsed_time = elapsed_time + simulator.t_step
        # Update the sketch
        update_canvas(x0, u0, elapsed_time)
        
    if event == 'run_manually':
        # In this case, start a simulator on a separate thread
        nr_iteration = 0
        max_iterations = int(values['nr_iter'])
        while nr_iteration < max_iterations:
            u0 = np.array(values['push_F']).reshape(1,1)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
            elapsed_time = elapsed_time + simulator.t_step
            nr_iteration += 1
            # Update the sketch
            update_canvas(x0, u0, elapsed_time)
            
    if event == 'one_step':
        tic = time.time()
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)
        toc = time.time()
        elapsed_time = elapsed_time + simulator.t_step
        # Update the sketch
        update_canvas(x0, u0, elapsed_time)
        
    if event == 'run_nmpc':
        nmpc_iteration =0
        max_iterations = int(values['nr_iter'])
        while nmpc_iteration < max_iterations:
            tic = time.time()
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = estimator.make_step(y_next)
            toc = time.time()
            time_list.append(toc-tic)
            nmpc_iteration += 1
            elapsed_time = elapsed_time + simulator.t_step

            # Update the sketch
            update_canvas(x0, u0, elapsed_time)

window.close()


