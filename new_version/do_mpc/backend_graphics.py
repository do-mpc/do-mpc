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


class backend_graphics:
    def __init__(self):
        self.line_list = []


    def add_line(self, var_type, var_name, axis):
        self.line_list.append(
            {'var_type': var_type, 'var_name': var_name, 'ax': axis}
        )


    def plot_results(self, data):

        lines = []
        for line_i in self.line_list:
            time = data._time
            res_type = getattr(data, line_i['var_type'])
            # The .f() method returns an index of a casadi Struct, given a name.
            var_ind = getattr(data.model, line_i['var_type']).f[line_i['var_name']]
            lines.extend(line_i['ax'].plot(time, res_type[:, var_ind]))

        return lines

    def plot_predictions(self, t_now, t_step, n_horizon, opt_x_num):
        lines = []
        for line_i in self.line_list:
            if line_i['var_type'] == '_x':
                t_end = t_now + (n_horizon+1)*t_step
                time = np.linspace(t_now, t_end, n_horizon+1)-t_step
            else:
                t_end = t_now + n_horizon*t_step
                time = np.linspace(t_now, t_end, n_horizon)-t_step

            if line_i['var_type'] in ['_x', '_z']:
                pred = vertcat(*opt_x_num[line_i['var_type'], :, 0, 0, line_i['var_name']])
            elif line_i['var_type'] in ['_u']:
                pred = vertcat(*opt_x_num[line_i['var_type'], :, 0, line_i['var_name']])

            lines.extend(line_i['ax'].plot(time, pred))

    def reset_axes(self, ax):
        for ax_i in ax:
            ax_i.lines = []
