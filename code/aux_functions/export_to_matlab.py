# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 17:38:16 2013

@author: Sergio Lucia
"""

# This script saves the results to a .txt file that can be loaded using Matlab

import scipy.io
n_alg_chosen = 1
#index_mpc = index_mpc - 1
data_simulation = NP.resize(NP.array([]),(index_mpc,1+nx+nu+n_alg_chosen))
data_simulation[:,0] = mpc_time[:index_mpc]
data_simulation[:,1:nx+1] = mpc_states[:index_mpc,:]
data_simulation[:,1+nx:1+nx+nu] = mpc_control[:index_mpc,:]*u_scaling
data_simulation[:,1+nx+nu:1+nx+nu+1] = NP.array([mpc_cost[:index_mpc]]).T  #Q_r
#data_simulation[:,1+nx+nu+1:1+nx+nu+2] = NP.array([mpc_alg[:index_mpc,8]]).T  #Mon_concentration
scipy.io.savemat('Classical2_n_robust'+str(n_robust)+'p_real'+str(int(10*p_real[0]))+'k_bias'+str(int(10*k_bias))+'setpoint.mat', mdict={'Classical2_n_robust'+str(n_robust)+'p_real'+str(int(10*p_real[0]))+'k_bias'+str(int(10*k_bias))+'setpoint':data_simulation})
