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
from casadi import *
from casadi.tools import *
import pdb

class configuration:
    def __init__(self, simulator, optimizer, estimator, x0=None):
        self.simulator = simulator
        self.optimizer = optimizer
        self.estimator = estimator

        if x0:
            # Set global intial condition.
            self.simulator.data.update('_x', x0)
            self.optimizer.data.update('_x', x0)
            self.estimator.data.update('_x', x0)
        else:
            # Set individual initial condition.
            self.simulator.data.update(_x = self.simulator._x0)
            self.optimizer.data.update(_x = self.optimizer._x0)
            self.estimator.data.update(_x = self.estimator._x0)

        self.simulator.data.update(_time = self.simulator._t0)
        self.optimizer.data.update(_time = self.optimizer._t0)
        self.estimator.data.update(_time = self.estimator._t0)


    def make_step_optimizer(self):
        x0 = self.optimizer._x0
        u_prev = self.optimizer._u0
        tvp0 = self.optimizer.tvp_fun(self.optimizer._t0)
        p0 = self.optimizer.p_fun(self.optimizer._t0)

        self.optimizer.opt_p_num['_x0'] = x0
        self.optimizer.opt_p_num['_u_prev'] = u_prev
        self.optimizer.opt_p_num['_tvp'] = tvp0['_tvp']
        self.optimizer.opt_p_num['_p'] = p0['_p']
        self.optimizer.solve()

        u0 = self.optimizer.opt_x_num['_u', 0, 0]
        z0 = self.optimizer.opt_x_num['_z', 0, 0, 0]

        # self.optimizer.data.update(_tvp = tvp0)
        # self.optimizer.data.update(_p = p0)
        self.optimizer.data.update(_u = u0)
        self.optimizer.data.update(_z = z0)

        self.optimizer._u0 = u0
        self.optimizer._z0 = z0

        t0 = self.optimizer._t0 = self.optimizer._t0 + self.optimizer.t_step
        self.optimizer.data.update(_time = t0)


    def make_step_simulator(self):
        tvp0 = self.simulator.tvp_fun(self.simulator._t0)
        p0 = self.simulator.p_fun(self.simulator._t0)
        x0 = self.simulator._x0
        u0 = self.optimizer._u0
        z0 = self.optimizer._z0

        self.simulator.sim_x_num['_x'] = x0
        self.simulator.sim_x_num['_z'] = z0
        self.simulator.sim_p_num['_u'] = u0
        self.simulator.sim_p_num['_p'] = p0
        self.simulator.sim_p_num['_tvp'] = tvp0

        self.simulator.simulate()

        x_next = self.simulator.sim_x_num['_x']
        z0 = self.simulator.sim_x_num['_z']

        self.simulator.data.update(_tvp = tvp0)
        self.simulator.data.update(_p = p0)
        self.simulator.data.update(_u = u0)
        self.simulator.data.update(_z = z0)
        self.simulator.data.update(_x = x_next)

        self.simulator._x0 = x_next

        t0 = self.simulator._t0 = self.simulator._t0 + self.simulator.t_step
        self.simulator.data.update(_time = t0)

    def make_step_estimator(self):
        # Usually a more complex mapping:
        self.estimator._x0 = self.simulator._x0

        self.optimizer._x0 = self.estimator._x0
