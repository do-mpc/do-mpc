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
        None

        self.axes_list = []


    def add_axes(self, var_type, var_name, **kwargs):
        self.axes_list.append([
            {'var_type': var_type, 'var_name': var_name, **kwargs}
        ])

    def setup_plot(self):
        n_subplots = len(self.axes_list)
        fig, ax = plt.subplots(n_subplots, sharex=True)

        for ax_i in ax:
            ax_i.set_xlabel('time')
            ax_1.set_ylabel(self.axes_list['var_name'])

    def plot_prediction(self, t_now, data, opt_x_num):
        None
