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
import time
rel_do_mpc_path = os.path.join('..')
sys.path.append(rel_do_mpc_path)
import do_mpc
import pickle
# %%



# %%


""" User settings: """
show_animation = True
store_results = False

"""
Get configured do-mpc modules:
"""

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator
#from template_mhe import template_mhe
import pdb

model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
#mhe = template_mhe(model)


"""
Set initial state
"""
np.random.seed(99)

# Use different initial state for the true system (simulator) and for MHE / MPC
x0 = np.array([[0],[0],[10],[0],[1/3]])+(np.random.rand(model.n_x)-0.5).reshape((5,1))
x0[4]=0.1
x0[2]=10
x0_true = np.array([[0],[0],[20],[0],[1/3]])#np.ones(model.n_x)

mpc.x0 = x0
simulator.x0 = x0_true
#mhe.x0 = x0
#mhe.p_est0 = 1e-4

# Set initial guess for MHE/MPC based on initial state.
mpc.set_initial_guess()
#mhe.set_initial_guess()

"""
Setup graphic:
"""

color = plt.rcParams['axes.prop_cycle'].by_key()['color']

with open('data_scenario_1.pickle', 'rb') as f:
    data = pickle.load(f)
fig, ax = plt.subplots(5,1, sharex=True, figsize=(10, 9))

mpc_plot = do_mpc.graphics.Graphics(mpc.data)
#mhe_plot = do_mpc.graphics.Graphics(mhe.data)
sim_plot = do_mpc.graphics.Graphics(simulator.data)

ax[0].set_title('Positions:')
#mhe_plot.add_line('_x', 'y', ax[0])
#mhe_plot.add_line('_tvp', 'y_set', ax[0], color=color[0], linestyle='--', alpha=0.5)
#mhe_plot.add_line('_x', 'x', ax[0],color=color[1])
#ax[0].legend(
#    mhe_plot.result_lines['_x', 'y']+mhe_plot.result_lines['_tvp', 'y_set']+mhe_plot.pred_lines['_x', 'y']+mhe_plot.result_lines['_x','x'],
#    ['Recorded y', 'Setpoint y', 'Predicted y','Recorded x'], title='Positions')

#ax[1].set_title('Velocity:')

#mhe_plot.add_line('_x', 'v', ax[1])

#ax[1].legend(
#    mhe_plot.result_lines['_x', 'v'],
#    [ 'v [m/s]']
#    )

#ax[2].set_title('Inputs:')
#mhe_plot.add_line('_u', 'a', ax[2])
#mhe_plot.add_line('_u', 'delta_f', ax[2])

ax[3].set_title('Estimated angle:')
sim_plot.add_line('_x', 'psi', ax[3])
#mhe_plot.add_line('_x', 'psi', ax[3])


ax[4].set_title('Estimated parameters:')
sim_plot.add_line('_x', 'cog', ax[4])
#mhe_plot.add_line('_x', 'cog', ax[4])

#for mhe_line_i, sim_line_i in zip(mhe_plot.result_lines.full, sim_plot.result_lines.full):
#    mhe_line_i.set_color(sim_line_i.get_color())
#    sim_line_i.set_alpha(0.5)
#    sim_line_i.set_linewidth(5)

ax[0].set_ylabel('x/y \n [m]')
ax[1].set_ylabel('v \n [m/s]')
ax[2].set_ylabel('acceleration \n [m/s^2]')
ax[3].set_ylabel('angle\n [rad]')
ax[4].set_ylabel('Cog')
ax[3].set_xlabel('time [s]')

for ax_i in ax:
    ax_i.axvline(1.0)

fig.tight_layout()
plt.ion()

"""
Run MPC main loop:
"""
# ekf start
# ekf initilisation
nx=model.n_x
nu=model.n_u
Q=np.array([[1e-1],[1e-1],[1e-2],[1e-4],[1e-9]])*np.eye(nx)
R=np.array([[1e-1],[1e-1],[1e-4],[1e-4]])*np.eye(model.n_v)
P_0=np.array([[1],[1],[10],[1],[1]])*np.eye(nx)#Q

x, u, z, tvp, p, w = model['x', 'u', 'z', 'tvp', 'p', 'w']
up=model['u','p']

f=model._rhs
h=model._y_expression
df_dx = jacobian(f, x)
df_du = jacobian(f, u)
# Create symbolic expression for the derivative of the measurement equations w.r.t to the states, inputs
dh_dx = jacobian(h, x)
dh_du = jacobian(h, u)
df_dx_fun = Function('df_dx_fun', [x, u, z, tvp, p, w], [df_dx])
df_du_fun = Function('df_du_fun', [x, u, z, tvp, p, w], [df_du])
dh_dx_fun = Function('dh_dx_fun', [x, u, z, tvp, p, w], [dh_dx])
dh_du_fun = Function('dh_du_fun', [x, u, z, tvp, p, w], [dh_du])
f_fun=Function('f_fun',[x, u, z, tvp, p, w], [f])

x0_observer = x(x0)
u0 = u(0)
z0 = z(0)
tvp0 = tvp(0)
tvp0['P_v']=np.diag(np.array([1e1,1e1,1e4,1e4]))
p0 = p(0)
#p0['cog']=1/3
#p0['P_p']=10
w0 = w(0)


int_states = struct_symSX([
        entry('P', shape=(5, 5)),
        entry('x',shape=(5,1))
])

f_new = model._rhs_fun(int_states['x'], model.u, model.z, model.tvp, model.p, 0)
df_dx_new = df_dx_fun(int_states['x'], model.u, model.z, model.tvp, model.p, 0)

rhs_states =struct_SX(int_states)
rhs_states['P'] =df_dx_new@int_states['P']+ int_states['P']@ df_dx_new.T+Q
rhs_states['x']= f_new
opts={'tf': 0.1}
integrator_0 = integrator('test', 'cvodes', {'x': int_states, 'ode': rhs_states,'p':vertcat(u,p)}, opts)

meas_array=np.zeros((200,4))
x0_list=[np.resize(x0,(5,1))[4]]
x0_observer_list=[np.array(x0_observer.cat)[4]]
for k in range(200):

    u0 = data[k,:].reshape((model.n_u,1))# mpc.make_step(x0)


    int_states_0 = int_states(0)
    int_states_0['P']=P_0
    int_states_0['x']=x0_observer
    d=np.array([[float(u0[0])],[float(u0[1])],[1]])

    sol, _, _, _, _, _ = integrator_0(int_states_0, d, 0, 0, 0,0)
    P_0=sol.full()[0:model.n_x**2].reshape((model.n_x,model.n_x))
    x0_observer=sol.full()[model.n_x**2:model.n_x**2+model.n_x]
    x0_observer=x(x0_observer)
    y_next = simulator.make_step(u0, v0=np.array([[1e-1],[1e-1],[1e-4],[1e-4]])*np.random.randn(model.n_v,1),w0=np.array([[1e-1],[1e-1],[1e-2],[1e-4],[0e-1]])*np.random.randn(model.n_w,1))
    meas_array[k] = y_next.squeeze()
    #print(y_next)
    C_tilde = dh_dx_fun(x0_observer, u0, z0, tvp0, p0, w0)
    K = P_0 @ C_tilde.T @ inv(C_tilde @ P_0 @ C_tilde.T + R)
    P_0 = (np.eye(nx) - K @ C_tilde) @ P_0
    x0_observer=x(x0_observer.cat+K @ (y_next - C_tilde @ x0_observer.cat))
    #x0 = mhe.make_step(y_next)
    x0_observer_list.append(x0_observer.cat.full()[4])
    x0_list.append(x0[4])

    if show_animation:
        #mpc_plot.plot_results()
        #mpc_plot.plot_predictions()
        #mhe_plot.plot_results()
        sim_plot.plot_results()

        #mpc_plot.reset_axes()
        #mhe_plot.reset_axes()
        sim_plot.reset_axes()
        plt.show()
        plt.pause(0.01)

plt.figure()
x0_list=np.asarray(x0_list)
x0_observer_list=np.asarray(x0_observer_list)
plt.plot(x0_list)
plt.plot(x0_observer_list)
plt.show()
fig,ax=plt.subplots()
#ax.plot(mhe.data['_x','x'],mhe.data['_x','y'])
plt.show()
#plt.savefig('trajectory.svg')
data=dict()
data['x']=meas_array[:,0]
data['y']=meas_array[:,1]
data['a']=meas_array[:,2]
data['delta_f']=meas_array[:,3]
with open('data_scenario_1.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


input('Press any key to exit.')

# Store results:
#if store_results:
#    do_mpc.data.save_results([mpc, mhe, simulator], 'rot_oscillating_masses')