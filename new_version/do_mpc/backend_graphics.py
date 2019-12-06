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

    def reset_axes(self):
        for ax_i in self.ax_list:
            ax_i.lines = []
        self.reset_prop_cycle()

    def reset_prop_cycle(self):
        for ax_i in self.ax_list:
            ax_i.set_prop_cycle(None)

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

    def plot_predictions(self, data, opt_x_num=None):
        assert data.dtype == 'optimizer', 'Can only call plot_predictions with data object from do-mpc optimizer.'

        t_now = data._time[-1]
        # These fields only exist, if data type (dtype) os optimizer:
        t_step = data.meta_data['t_step']
        n_horizon = data.meta_data['n_horizon']
        structure_scenario = data.meta_data['structure_scenario']

        # Check if full solution is stored in data, or supplied as optional input. Raise error is neither is the case.
        if opt_x_num is None and data.meta_data['store_full_solution']:
            opt_x_num = data.opt_x(data._opt_x_num[-1])
        elif opt_x_num is not None:
            pass
        else:
            raise Exception('Cannot plot predictions if full solution is not stored or supplied when calling the method.')

        # Plot predictions:
        self.reset_prop_cycle()
        lines = []
        for line_i in self.line_list:
            # Fix color for the robust trajectories according to the current state of the cycler.
            if 'color' not in line_i['pltkwargs']:
                color = next(line_i['ax']._get_lines.prop_cycler)['color']
                line_i['pltkwargs'].update({'color':color})


            # Choose time array depending on variable type (states with n+1 steps)
            if line_i['var_type'] in ['_x', '_z']:
                t_end = t_now + (n_horizon+1)*t_step
                time = np.linspace(t_now, t_end, n_horizon+1)
            else:
                t_end = t_now + n_horizon*t_step
                time = np.linspace(t_now, t_end, n_horizon)
                structure_scenario = structure_scenario[:-1,:]

            # Plot states etc. as continous quantities and inputs as steps.
            if line_i['var_type'] in ['_x', '_z']:
                # pred is a n_horizon x n_branches array.
                pred = vertcat(*opt_x_num[line_i['var_type'],:,lambda v: horzcat(*v),:, 0, line_i['var_name']])
                # sort pred such that each column belongs to one scenario
                pred = pred.full()[range(pred.shape[0]),structure_scenario.T].T
                lines.extend(line_i['ax'].plot(time, pred, **line_i['pltkwargs']))
            elif line_i['var_type'] in ['_u']:
                pred = vertcat(*opt_x_num[line_i['var_type'],:,lambda v: horzcat(*v),:,line_i['var_name']])
                pred = pred.full()[range(pred.shape[0]),structure_scenario.T].T
                lines.extend(line_i['ax'].step(time, pred, **line_i['pltkwargs']))
