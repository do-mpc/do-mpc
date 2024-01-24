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
from ._base import Estimator
import do_mpc
from typing import Union
import casadi as ca



class EKF(Estimator):
    """Extended Kalman Filter. Setup this class and use :py:func:`EKF.make_step`
    during runtime to obtain the currently estimated states given the measurements ``y0``.

    Warnings:
        Work in progress.
    """
    def __init__(self, model:Union[do_mpc.model.Model,do_mpc.model.LinearModel], Q, R):
        
        # init
        Estimator.__init__(self, model)
        #self.similator = sim
        #self.flags = model.flags
        #self.model = model

        # number of states and outputs
        self.nx = model.x.shape[0]
        self.ny = model.y.shape[0]

        # initialises the initial guess of the estimator
        self.x0 = np.empty((self.nx, 1))
        self.x_ekf = np.empty((self.nx, 1))

        # Stores the variances of state(s) and measurement(s)
        self.Q = Q
        self.R = R

        # Covariance Matrix of initial error
        self.P0 = Q
        self.counter = 0

        # generating flags
        self.flags = {
            'setup': True,
            'set_initial_guess': False,
        }

    def set_initial_guess(self):

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'

        # set initial value
        self.x_ekf = ca.DM(self.x0).full()

        # changing flag
        self.flags['set_initial_guess'] = True

        # return
        return None

    def make_step(self, y_next, u_next, simulator):
        #print("Iternation No:", self.counter)
        self.counter += 1

        # checks to ensure proper usage
        assert self.flags['setup'] == True, 'EKF was not setup yet. Please call EKF.setup().'
        assert self.flags['set_initial_guess'] == True, 'Initial guess was not provided. Please call EKF.set_initial_guess().'

        # Measurement update / Correction
        # needs to be checked
        #A,B,C,D = self.model.get_linear_system_matrices(xss = self.x_ekf, uss = u_next, z = self.model.z, tvp = self.model.tvp, p = self.model.p)
        A,B,C,D = self.model.get_linear_system_matrices(xss = self.x_ekf, uss = u_next, z = simulator.sim_z_num['_z'].full(), tvp = simulator.sim_p_num['_tvp'].full(), p = simulator.sim_p_num['_p'].full())

        A = ca.DM(A).full()
        B = ca.DM(B).full()
        C = ca.DM(C).full()
        D = ca.DM(D).full()
        
        # Observability 
        assert self.check_obsevability(A,C), 'System not observable. EKF failed!'

        # Optimal Kalman gain
        denominator = C @ self.P0 @ C.T + self.R
        L = self.P0 @ C.T @ ca.DM(ca.inv_minor(denominator)).full()
        
        # Updating observer with Kalman filter (Aposteriori)
        x_ekf_current = self.x_ekf + L @ (y_next - C @ self.x_ekf)
        self.x_ekf = ca.DM(x_ekf_current).full()

        # Updating error covariance matrix aposteriori
        self.P0 = ca.DM((np.eye(self.nx) - L @ C) @ self.P0).full()

        # Prediction (Apriori)
        self.x_ekf = ca.DM((A@self.x_ekf) + (B@u_next)).full()

        # Updating error covariance matrix of apriori
        self.P0 = ca.DM(A @ self.P0 @ A.T + self.Q).full()

        # return
        return x_ekf_current
    

    def check_obsevability(self, A, C):
        """
        obs_mat = C
        print("C:", C)
        for i in range(self.nx-1):
            CAN = C @ np.linalg.matrix_power(A, i+1)
            print(i)
            print(CAN)
            obs_mat = np.vstack([obs_mat, CAN])

        print(obs_mat.shape)
        """
        return True