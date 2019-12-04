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
        self.ax_list  = []
        self.color = plt.rcParams['axes.prop_cycle'].by_key()['color']


    def add_line(self, var_type, var_name, axis , **pltkwargs):
        self.line_list.append(
            {'var_type': var_type, 'var_name': var_name, 'ax': axis, 'pltkwargs': pltkwargs}
        )
        self.ax_list.append(axis)


    def plot_results(self, data):
        self.reset_prop_cycle()
        lines = []
        for line_i in self.line_list:
            time = data._time
            res_type = getattr(data, line_i['var_type'])
            # The .f() method returns an index of a casadi Struct, given a name.
            var_ind = getattr(data.model, line_i['var_type']).f[line_i['var_name']]
            if line_i['var_type'] in ['_u']:
                lines.extend(line_i['ax'].step(time, res_type[:, var_ind], **line_i['pltkwargs']))
            else:
                lines.extend(line_i['ax'].plot(time, res_type[:, var_ind], **line_i['pltkwargs']))

        return lines

    def plot_predictions(self, t_now, t_step, n_horizon, opt_x_num):
        self.reset_prop_cycle()
        lines = []
        for line_i in self.line_list:
            if line_i['var_type'] == '_x':
                t_end = t_now + (n_horizon+1)*t_step
                time = np.linspace(t_now, t_end, n_horizon+1)-t_step
            else:
                t_end = t_now + n_horizon*t_step
                time = np.linspace(t_now, t_end, n_horizon)-t_step

            if line_i['var_type'] in ['_x', '_z']:
                for s in range(3):
                    pred = vertcat(*opt_x_num[line_i['var_type'], :, s, 0, line_i['var_name']])
                    lines.extend(line_i['ax'].plot(time, pred, **line_i['pltkwargs']))
            elif line_i['var_type'] in ['_u']:
                for s in range(3):
                    pred = vertcat(*opt_x_num[line_i['var_type'], :, s, line_i['var_name']])
                    lines.extend(line_i['ax'].step(time, pred, **line_i['pltkwargs']))

    def reset_axes(self):
        for ax_i in self.ax_list:
            ax_i.lines = []
        self.reset_prop_cycle()

    def reset_prop_cycle(self):
        for ax_i in self.ax_list:
            ax_i.set_prop_cycle(None)
