# Wind model from 2017 Costello (Appendix A2)
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from scipy.signal import TransferFunction as TF
from casadi import *
import pdb

T_end = 800         # [s] length of wind trajectory to be generated
N_traj = 3

# given parameters
w_ref = 7.0 + 2.0*np.random.uniform()           # [m/s]
z_ref = 10          # [m]
L_W = 100           # [m]
k_sigma_w = 0.14    # [-]
Chi = 15            # [°]
a = 0.15            # [-] surface friction coefficient
T_W = 0.15           # [s] update period of w_N

# static parameters
tau_F = L_W / w_ref
K_F = np.sqrt(1.49*tau_F/T_W)
sigma_w = k_sigma_w * w_ref         # standard deviation of w_N
bar_w_N = -sigma_w/2/w_ref

num = [K_F]
den = [tau_F,1.0]
H_F_tf_cont = TF(num,den)
H_F_ss_cont = H_F_tf_cont.to_ss()
A_cont = float(H_F_ss_cont.A)
B_cont = float(H_F_ss_cont.B)
C_cont = float(H_F_ss_cont.C)
w = SX.sym("w")
x = SX.sym("x")
u = SX.sym("u")
ode = (A_cont * x + B_cont * u) * k_sigma_w * w_ref
dae = {'x':w,'p':vertcat(x,u), 'ode':ode}
ode_f = A_cont * x + B_cont * u
dae_f = {'x':x,'p':u, 'ode':ode_f}
opts = {"abstol":1e-10,"reltol":1e-10, 'tf':T_W}
simulator_f = integrator("simulator_f", "cvodes", dae_f, opts)
simulator = integrator("simulator", "cvodes", dae, opts)

H_F_ss_dis = H_F_ss_cont.to_discrete(dt=T_W)
A_dis = float(H_F_ss_dis.A)
B_dis = float(H_F_ss_dis.B)
C_dis = float(H_F_ss_dis.C)

# generate wind trajectory
w_N_dis = []
w_dis = []
w_N_cont = []
w_cont = []
for j in range(N_traj):
    f_init = np.random.normal()/4
    f_init_dis = f_init
    f_init_cont = f_init
    w_init = w_ref + bar_w_N + sigma_w * C_cont * f_init_cont
    w_N_cur_dis = []
    w_cur_dis = []
    w_N_cur_cont = []
    w_cur_cont = []
    for i in range(int(T_end/T_W)+1):
        w_N_cur_dis.append(bar_w_N + sigma_w * C_dis * f_init_dis)
        w_cur_dis.append(w_ref + w_N_cur_dis[-1])
        # w_N_cur_cont.append(bar_w_N + sigma_w * C_cont * f_init_cont)
        # w_cur_cont.append(w_ref + w_N_cur_cont[-1])
        w_cur_cont.append(w_init)

        new_u = np.random.normal()/4
        f_init_dis = A_dis * f_init_dis + B_dis * new_u
        w_init = float(simulator(x0 = w_init, p = vertcat(f_init_cont,new_u))['xf'])
        f_init_cont = float(simulator_f(x0 = f_init_cont, p = new_u)['xf'])


    # plot wind
    plt.figure()
    plt.grid()
    plt.plot(w_cur_dis,label='dis')
    plt.plot(w_cur_cont,label='cont')
    plt.legend()

input('bla')
