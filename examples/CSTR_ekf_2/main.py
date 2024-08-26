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
import casadi as ca
from casadi.tools import *
#import pdb
import sys
import os
rel_do_mpc_path = os.path.join('..','..')
sys.path.append(rel_do_mpc_path)
import do_mpc
from do_mpc.tools import Timer

import matplotlib.pyplot as plt
#import pickle
#import time


from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator

""" User settings: """
show_animation = False
store_results = False


model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
#ekf = template_ekf(model)
estimator = do_mpc.estimator.StateFeedback(model)

# setting up model variances with a generic value
#q = 1e-3 * np.ones(model.n_x)
#q = 1000 * np.ones(model.n_x)
#r = 1e-2 * np.ones(model.n_y)
#r = 100 * np.ones(model.n_y)

#Q = np.diag(q.flatten())
#R = np.diag(r.flatten())

# Set the initial state of mpc and simulator:
C_a_0 = 0.8 # This is the initial concentration inside the tank [mol/l]
C_b_0 = 0.5 # This is the controlled variable [mol/l]
T_R_0 = 134.14 #[C]
T_K_0 = 130.0 #[C]
x0 = np.array([C_a_0, C_b_0, T_R_0, T_K_0]).reshape(-1,1)

mpc.x0 = x0
simulator.x0 = x0
#ekf.x0 = x0

mpc.set_initial_guess()
#ekf.set_initial_guess()

"""
# Initialize graphic:
graphics = do_mpc.graphics.Graphics(mpc.data)


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
for line_i in graphics.pred_lines.full:
    line_i.set_linewidth(1)

label_lines = graphics.result_lines['_x', 'C_a']+graphics.result_lines['_x', 'C_b']
ax[0].legend(label_lines, ['C_a', 'C_b'])
label_lines = graphics.result_lines['_x', 'T_R']+graphics.result_lines['_x', 'T_K']
ax[1].legend(label_lines, ['T_R', 'T_K'])

fig.align_ylabels()
fig.tight_layout()
plt.ion()

"""

x_data = [x0]
x_hat_data = [x0]

timer = Timer()

# ==============EKF_Setup==============
# setting up continious ekf
t_step = 0.005
n_sim= 50
nx=model.n_x
nu=model.n_u
q = 1e-3 * np.ones(model.n_x)
r = 1e-2 * np.ones(model.n_y)
Q = np.diag(q.flatten())
R = np.diag(r.flatten())
ekf_parameters = np.array([1, 1]).reshape(-1,1)


# extracting & setting up symbolic variables
x, u, z, tvp, p, w = model['x', 'u', 'z', 'tvp', 'p', 'w']
up = model['u', 'p']

# state space rhs
f = model._rhs
f_fun = Function('f_fun',[x, u, z, tvp, p, w], [f])

# measurement rhs
h = model._y_expression

# A matirx in ss
df_dx = jacobian(f, x)
df_dx_fun = Function('df_dx_fun', [x, u, z, tvp, p, w], [df_dx])

# B matirx in ss
df_du = jacobian(f, u)
df_du_fun = Function('df_du_fun', [x, u, z, tvp, p, w], [df_du])

# Create symbolic expression for the derivative of the measurement equations w.r.t to the states, inputs
# C matrix in ss
dh_dx = jacobian(h, x)
dh_dx_fun = Function('dh_dx_fun', [x, u, z, tvp, p, w], [dh_dx])

# D matrix in ss
dh_du = jacobian(h, u)
dh_du_fun = Function('dh_du_fun', [x, u, z, tvp, p, w], [dh_du])

# initialising
P_0 = Q
# Here the argument passed inside does not denote the iteration number. It is quite literally the current value itself. This is someting like a 'casadi.function'.
x0_observer = x(x0)
u0 = u(0)
z0 = z(0)
tvp0 = tvp(0)
p0 = p(ekf_parameters)
w0 = w(0)

# initilising struct for convenient passage to integrator
#int_states = struct_symSX([
#        entry('P', shape=(nx, nx)),
#        entry('x', shape=(nx, 1))
#])

# covariance
P = SX.sym('P', nx, nx)
dP_dt = (df_dx @ P) + (P @ df_dx.T) + Q
dP_dt_fun = Function('df_dx_fun', [P, x, u, z, tvp, p, w], [dP_dt])


# accounting for parameter estimation
#f_new = model._rhs_fun(int_states['x'], model.u, model.z, model.tvp, model.p, 0)
#df_dx_new = df_dx_fun(int_states['x'], model.u, model.z, model.tvp, model.p, 0)

#rhs_states =struct_SX(int_states)
#rhs_states['P'] =df_dx_new@int_states['P']+ int_states['P']@ df_dx_new.T+Q
#rhs_states['x']= f_new
#rhs_states['P'] = (df_dx @ int_states['P']) + (int_states['P'] @ df_dx.T) + Q
#rhs_states['x'] = f

# integraion time step
#opts={'tf': 0.1}
opts = {'tf': t_step}
#integrator_0 = integrator('test', 'cvodes', {'x': int_states, 'ode': rhs_states,'p':vertcat(u,p)}, opts)
#integrator_0 = integrator('integrator_0', 'cvodes', {'x': int_states, 'ode': rhs_states, 'p': vertcat(u, p)}, opts)
#ode = {'x': vertcat(P, x), 'ode': vertcat(dP_dt_fun, f_fun), 'p': vertcat(u, p)}
#states = vertcat(P, x)
#equations = vertcat(dP_dt, f)
#ode = {'x': states, 'ode': equations}
covariance_simulator = integrator('covariance_simulator', 'cvodes', {'x': ca.reshape(P, -1, 1), 'ode': ca.reshape(dP_dt, -1, 1), 'p': ca.vertcat(x, p, u, tvp, z, w)}, opts)
#state_simulator = integrator('state_simulator', 'cvodes', {'x': x, 'ode': f}, opts)
#integrator_0 = integrator('integrator_0', 'cvodes', ode, opts)

# storage
meas_array = np.zeros((n_sim, nx))
x0_list = [np.resize(x0, (nx, 1))[nx - 1]]
x0_observer_list = [np.array(x0_observer.cat)[nx - 1]]

# main loop
for k in range(50):
    timer.tic()
    u0 = mpc.make_step(x0)
    timer.toc()
    #y_next = simulator.make_step(u0, v0=0.001*np.random.randn(model.n_v,1))
    #x0 = ekf.make_step(y_next = y_next, u_next = u0, Q=Q, R=R)

    # ekf_loop
    
    #int_states_0 = int_states(0)
    #int_states_0['P']=P_0
    #int_states_0['x']=x0_observer
    #d=np.array([[float(u0[0])],[float(u0[1])],[1],[1]])

    # Predict state ahead
    #sol, _, _, _, _, _, _ = integrator_0(int_states_0, 0, d, 0, 0,0,0)
    sol = covariance_simulator(x0 = P_0.reshape((-1, 1)), p =ca.vertcat(x0_observer, p0, u0, tvp0, z0, w0))

    # extracting Covariance ahead for the integrator output
    P_0=sol['xf'].full()[0:model.n_x**2].reshape((model.n_x,model.n_x))

    # extracting
    x0_observer=sol.full()[model.n_x**2:model.n_x**2+model.n_x]

    # ????
    x0_observer=x(x0_observer)

    # simulating next step
    y_next = simulator.make_step(u0)#, v0=np.array([[1e-1],[1e-1],[1e-4],[1e-4]])*np.random.randn(model.n_v,1),w0=np.array([[1e-1],[1e-1],[1e-2],[1e-4],[0e-1]])*np.random.randn(model.n_w,1))

    # ????
    meas_array[k] = y_next.squeeze()
    #print(y_next)

    # extracting measurement matrix
    C_tilde = dh_dx_fun(x0_observer, u0, z0, tvp0, p0, w0)

    # Kalman gain
    K = P_0 @ C_tilde.T @ inv(C_tilde @ P_0 @ C_tilde.T + R)

    # Update error covariance matrix
    P_0 = (np.eye(nx) - K @ C_tilde) @ P_0

    # Update estimate with meas
    x0_observer=x(x0_observer.cat+K @ (y_next - C_tilde @ x0_observer.cat))
    #x0 = mhe.make_step(y_next)

    # storage
    x0_observer_list.append(x0_observer.cat.full()[nx-1])
    #x0_list.append(x0[4])

    # storing datat for ekf plot
    x_data.append(simulator.data._x[-1].reshape((-1,1)))
    x_hat_data.append(x0)

    
    #x0 = estimator.make_step(y_next)
    # storing datat for ekf plot
    #x_data.append(simulator.data._x[-1].reshape((-1,1)))
    #x_hat_data.append(x0)


    if show_animation:
        graphics.plot_results(t_ind=k)
        graphics.plot_predictions(t_ind=k)
        graphics.reset_axes()
        plt.show()
        plt.pause(0.01)

timer.info()
timer.hist()


input('Press any key to exit.')
# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'CSTR_robust_MPC')

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
x_data = np.concatenate(x_data, axis=1)
x_hat_data = np.concatenate(x_hat_data, axis=1)
visualize(x_data, x_hat_data)

input('Press any key to exit.')